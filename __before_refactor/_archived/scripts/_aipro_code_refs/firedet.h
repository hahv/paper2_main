#pragma once

#include <iostream>
#include <deque>

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>

#include "global.h"
#include "model.h"

using namespace std;
using namespace cv;

/// Class for object detection
class FireDetector : public Model {
    bool fdEnable; /// flag to enable/disable object detection
    vector<int>* frameWidths;
    vector<int>* frameHeights;
    vector<float>* fpss;  // fpss are casted from double to int
    vector<float>* scaleFactors;

    int numChannels;
    int numClasses;

    int windowSize;
    vector<deque<int>> fireHistories;
    vector<deque<int>> smokeHistories;

    bool boostMode;
    int fdPeriod;
    vector<int> fdCnts;
    vector<bool> eagerEnables;

#ifndef _CPU_INFER
    // for temporal stabilization
    vector<Mat> tFrames; // temporal previous frames
    vector<Mat> deltaMasks;
    vector<int> minWs; // minimum width of the fire/smoke roi
    vector<int> minHs; // minimum height of the fire/smoke roi

    bool drawBlockDebug; // draw debugging info
    int colStep, rowStep; // step sizes for temporal stabilization
    bool temporalStabilizationEnable; // enable/disable temporal stabilization 

    bool temporalStabilization(Mat& frame, int vchID, Rect& roi);
#endif
public:
    FireDetector(Config& cfg);
    ~FireDetector() {};

    //for single frame
    bool runModel(FDRecord& fdRcd, Mat& frame, int vchID);
    bool runModelSingle(FDRecord& fdRcd, Mat& frame, int vchID);
};
