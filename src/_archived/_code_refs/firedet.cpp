#include <fstream>
#include <filesystem>
#include <numeric>

#include "firedet.h"

FireDetector::FireDetector(Config& cfg)
    : Model(cfg.fdEnable, cfg.igpuEnable, cfg.fdModelFile, 0, cfg.fdNetHeight, cfg.fdNetWidth, 3, 1) {
    fdEnable = cfg.fdEnable;

    if (!fdEnable)
        return;

    numClasses = cfg.fdNumClasses;
    numChannels = cfg.numChannels;

    frameWidths = &cfg.frameWidths;
    frameHeights = &cfg.frameHeights;
    fpss = &cfg.fpss;

    scaleFactors = &cfg.fdScaleFactors;

    windowSize = cfg.fdWindowSize;
    boostMode = cfg.boostMode;
    fdPeriod = cfg.fdPeriod;
    eagerEnables.resize(numChannels, true);
    fdCnts.resize(numChannels, 0);

    fireHistories.resize(numChannels);
    smokeHistories.resize(numChannels);

    for (int i = 0; i < numChannels; i++) {
        fireHistories[i].resize(windowSize, 0);
        smokeHistories[i].resize(windowSize, 0);
    }

#ifndef _CPU_INFER
    tFrames.resize(numChannels);
    deltaMasks.resize(numChannels);

    minWs.resize(numChannels, 0);
    minHs.resize(numChannels, 0);

    colStep = NET_WIDTH_FD / 4;
    rowStep = NET_HEIGHT_FD / 4;

    temporalStabilizationEnable = cfg.fdTemporalStabilization;
    drawBlockDebug = cfg.fdDrawBlockDebug;
#endif
}

bool FireDetector::runModelSingle(FDRecord& fdRcd, Mat& roiFrame, int vchID) {
    int fireFlag = 0, smokeFlag = 0;
    float* classScores;
    Mat preFrame;

    if (roiFrame.size() == Size(netWidth, netHeight))
        preFrame = roiFrame;
    else
        setPreFrame(preFrame, roiFrame);

#ifndef _CPU_INFER
    run(preFrame.ptr(), 0, numBindings - 1);
    classScores = (float*)hostBuffers[0];
#else
    vector<int> outputShape;
    classScores = run(preFrame.ptr(), outputShape);
#endif   

    int classID = 0;
    float maxScore = classScores[0];
    for (int i = 1; i < numClasses; i++) {
        if (classScores[i] > maxScore) {
            classID = i;
            maxScore = classScores[i];
        }
    }

    //float probs[NUM_FD_CLASSES] = { 0 };
    //float sumExp = 0.0f;
    //for (size_t i = 0; i < numClasses; ++i) {
    //    probs[i] = std::exp(classScores[i] - maxScore);
    //    sumExp += probs[i];
    //}

    if (classID == FD_CLASS_FIRE) {
        fireFlag = 1;
        smokeFlag = 1;
    }
    else if (classID == FD_CLASS_SMOKE)
        smokeFlag = 1;

    deque<int>& fireHistory = fireHistories[vchID];
    fireHistory.pop_front();
    fireHistory.push_back(fireFlag);

    float fireProb = (float)std::accumulate(fireHistory.begin(), fireHistory.end(), 0) / windowSize;
    fdRcd.fireProbs.pop_front();
    fdRcd.fireProbs.push_back(fireProb);

    deque<int>& smokeHistory = smokeHistories[vchID];
    smokeHistory.pop_front();
    smokeHistory.push_back(smokeFlag);

    float smokeProb = (float)std::accumulate(smokeHistory.begin(), smokeHistory.end(), 0) / windowSize;
    fdRcd.smokeProbs.pop_front();
    fdRcd.smokeProbs.push_back(smokeProb);

    if (!boostMode) {
        if (fireProb > 0 || smokeProb > 0)
            eagerEnables[vchID] = true;
        else
            eagerEnables[vchID] = false;
    }

    return true;
}

bool FireDetector::runModel(FDRecord& fdRcd, Mat& frame, int vchID) {
    if (!fdEnable)
        return true;

    if (!boostMode && !eagerEnables[vchID]) {
        fdCnts[vchID]++;

        if (fdCnts[vchID] < fdPeriod) {
            fdRcd.fireProbs.pop_front();
            fdRcd.fireProbs.push_back(0.0f);

            fdRcd.smokeProbs.pop_front();
            fdRcd.smokeProbs.push_back(0.0f);

            return true;
        }
    }

    fdCnts[vchID] = 0;

#ifndef _CPU_INFER
    Mat oriFrame;
    Rect roi;

    if (drawBlockDebug)
        oriFrame = frame.clone();
    else
        oriFrame = frame;

    bool doFDInference = temporalStabilization(frame, vchID, roi);

    if (!doFDInference) {
        fdRcd.fireProbs.pop_front();
        fdRcd.fireProbs.push_back(0.0f);

        fdRcd.smokeProbs.pop_front();
        fdRcd.smokeProbs.push_back(0.0f);
    }
    else {
        Mat roiFrame = oriFrame(roi);
		        
        if (fdRcd.exROI.area() > 0) {
            Rect overlap = roi & fdRcd.exROI;

            if (overlap.area() > 0) {
                roiFrame = roiFrame.clone();

				// Adjust overlap coordinates to the roi
                overlap.x -= roi.x;
                overlap.y -= roi.y;

				rectangle(roiFrame, overlap, Scalar(114, 114, 114), cv::FILLED); // Draw the overlap rectangle
            }
        }

        runModelSingle(fdRcd, roiFrame, vchID);
    }
#else
    runModelSingle(fdRcd, frame, vchID);
#endif

    return true;
}


#ifndef _CPU_INFER        
bool FireDetector::temporalStabilization(Mat& frame, int vchID, Rect& roi) {
    const int DIFF_FRAME_TH = 1; // difference threshold for frame comparison
    const int IMPACK_PLUS_ONE = 5; // difference threshold for frame comparison
    const int MASK_TH = 10; // threshold for acculurated changes in the mask
    const int ROI_TH = 200; // thereshold for detecting roi to be examined

    Mat& tFrame = tFrames[vchID];
    if (tFrame.empty() || frame.size() != tFrame.size()) {
        tFrame = frame.clone();
        roi = Rect(0, 0, frame.cols, frame.rows); // Initialize roi to the full frame
        return false;
    }

    int& minW = minWs[vchID];
    int& minH = minHs[vchID];

    Mat& deltaMask = deltaMasks[vchID];
    if (deltaMask.empty() || deltaMask.size() != frame.size()) {
        deltaMask = Mat::zeros(frame.size(), CV_8UC1);

        if (frame.cols == 1920 && frame.rows == 1080) {
            minW = 1280;
            minH = 720;
        }
        else {
            minW = 0.75 * frame.cols; //HD: 960, 540
            minH = 0.75 * frame.rows;
        }
    }

    Mat delta;
    absdiff(tFrame, frame, delta);
    cvtColor(delta, delta, COLOR_BGR2GRAY);
    threshold(delta, delta, DIFF_FRAME_TH, IMPACK_PLUS_ONE, THRESH_BINARY);

    deltaMask = deltaMask + delta;
    cv::min(deltaMask, 25, deltaMask);
    subtract(deltaMask, 1, deltaMask);

    cv::Mat curMask;
    cv::compare(deltaMask, MASK_TH, curMask, cv::CMP_GE);

    frame.copyTo(tFrame);

    int rows = curMask.rows;
    int cols = curMask.cols;
    int min_x0 = cols, min_y0 = rows, max_x1 = 0, max_y1 = 0;
    int nonZeroCountTotal = 0, nonZeroCountInside = 0;

    for (int row = 0; row < rows; row += rowStep) {
        for (int col = 0; col < cols; col += colStep) {
            int blockHeight = std::min(rowStep, rows - row);
            int blockWidth = std::min(colStep, cols - col);
            cv::Rect block(col, row, blockWidth, blockHeight);
            cv::Mat blockMask = curMask(block);
            int nonZeroCount = cv::countNonZero(blockMask);
            nonZeroCountTotal += nonZeroCount;

            if (nonZeroCount < ROI_TH)
                continue; // Skip this region if it has too few changes

            nonZeroCountInside += nonZeroCount;

            // Update the bounding box coordinates
            if (col < min_x0) min_x0 = col;
            if (row < min_y0) min_y0 = row;
            if (col + blockWidth > max_x1) max_x1 = col + blockWidth;
            if (row + blockHeight > max_y1) max_y1 = row + blockHeight;

            if (drawBlockDebug) {
                rectangle(frame, block, Scalar(0, 0, 255), 2); // Draw rectangle around the region

                //Mat overlay = frame(block).clone();
                //overlay.setTo(Scalar(255, 0, 0), regionMask);
                //double alpha = 0.7;
                //addWeighted(overlay, alpha, frame(block), 1.0 - alpha, 0, frame(block));            
                putText(frame, std::to_string(nonZeroCount),
                    Point(col + 10, row + 20),
                    FONT_HERSHEY_PLAIN, 1, Scalar(0, 255, 0), 1);
            }
        }
    }

    if (drawBlockDebug) {
        cout << nonZeroCountTotal << " "; // Debug: print the number of changed pixels
        putText(frame, std::to_string(nonZeroCountTotal),
            Point(10, 60), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);
    }

    // Draw a rectangle around the overall bounding box of all regions with significant changes
    if (min_x0 < max_x1 && min_y0 < max_y1) {
        Rect initROI(min_x0, min_y0, max_x1 - min_x0, max_y1 - min_y0);

        if (initROI.width < minW) {
            initROI.x = std::min(cols - minW, std::max(0, initROI.x - (minW - initROI.width) / 2)); // Center the ROI
            initROI.width = minW; // Ensure minimum width			
        }
        if (initROI.height < minH) {
            initROI.y = std::min(rows - minH, std::max(0, initROI.y - (minH - initROI.height) / 2)); // Center the ROI
            initROI.height = minH; // Ensure minimum height		    
        }

        roi = initROI;

        if (drawBlockDebug) {
            rectangle(frame, roi, Scalar(0, 255, 0), 2); // Draw rectangle around the overall region
            putText(frame, std::to_string(nonZeroCountInside),
                Point(min_x0 + 10, min_y0 + 40), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);
        }

        return true; // do fd inference
    }
    else
        return false; // skip fd inference
}
#endif
