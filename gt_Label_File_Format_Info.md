# Video Ground Truth (GT) Label File Formats

This document outlines the two supported Ground Truth (GT) label file formats for frame-level video annotations. Both formats map individual video frames to their corresponding classification labels, where the label is one of three text classes: **`fire`**, **`smoke`**, or **`none`**, and frame indexing is **1-based** (starts from `1`).

---

## 1. Space-Delimited Text Format (`.txt`)

A simple, lightweight format where annotations are stored in a plain text file named directly after the source video file (including its original extension).

### Naming Convention
* **Pattern:** `<video_name_with_extension>.txt`
* **Example:** `aihub__lb_fire__0182.mp4.txt` *(corresponding to `aihub__lb_fire__0182.mp4`)*

### Structure & Syntax
* **Delimiter:** Single ASCII space (` `)
* **Header:** None
* **Line Format:** `frame_idx gt_label`

### Schema

| Field | Type | Allowed Values | Description |
|---|---|---|---|
| **`frame_idx`** | Integer | `1`, `2`, `3`, ... | The sequential frame index within the video (1-indexed). |
| **`gt_label`** | String | `fire`, `smoke`, `none` | The ground truth class label assigned to the frame. |

### Example File (`aihub__lb_fire__0182.mp4.txt`)
```text
1 none
2 none
3 smoke
4 smoke
5 fire
6 fire
```

## 2. Tabular CSV Format (`.csv`)

A standard Comma-Separated Values (CSV) format that includes explicit column headers and embeds the source video path alongside each frame annotation. This format is ideal for multi-video datasets or pipelines requiring self-contained metadata.

### Naming Convention

- **Pattern:** `<video_id>__labels.csv`
- **Example:** `aihub__lb_fire__0182__labels.csv`

### Structure & Syntax

- **Delimiter:** Semicolon(`;`)
- **Header:** Required on line 1 (`frame_idx,video_path,label`)
- **Line Format:** `frame_idx,video_path,label`

### Schema

| **Field**        | **Type** | **Allowed Values**      | **Description**                                              |
| ---------------- | -------- | ----------------------- | ------------------------------------------------------------ |
| **`frame_idx`**  | Integer  | `1`, `2`, `3`, ...      | The sequential frame index within the video (1-indexed).     |
| **`video_path`** | String   | Any valid path          | The file name, relative path, or absolute path to the source video file. |
| **`label`**      | String   | `fire`, `smoke`, `none` | The ground truth class label assigned to the frame.          |

### Example File (`aihub__lb_fire__0182__labels.csv`)

Code snippet

```
frame_idx;video_path;label
1;aihub__lb_fire__0182.mp4;none
2;aihub__lb_fire__0182.mp4;none
3;aihub__lb_fire__0182.mp4;smoke
4;aihub__lb_fire__0182.mp4;smoke
5;aihub__lb_fire__0182.mp4;fire
6;aihub__lb_fire__0182.mp4;fire
```

## Format Comparison

| **Attribute**       | **Text Format (.txt)**  | **CSV Format (.csv)**              |
| ------------------- | ----------------------- | ---------------------------------- |
| **File Naming**     | `<video.ext>.txt`       | `<video_id>__labels.csv`           |
| **Delimiter**       | Space (` `)             | Semicolon (`;`)                    |
| **Header Row**      | No                      | Yes (`frame_idx;video_path;label`) |
| **Video Reference** | Implicit (via filename) | Explicit (via `video_path` column) |
| **Frame Indexing**  | 1-based (`1, 2, 3...`)  | 1-based (`1, 2, 3...`)             |
| **Allowed Labels**  | `fire`, `smoke`, `none` | `fire`, `smoke`, `none`            |