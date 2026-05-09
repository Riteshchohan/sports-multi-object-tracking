---
(You can expand this later — basic template is ready)

```markdown
# Technical Report

## 1. Objective
Implemented multi-object detection and persistent ID tracking on public sports video.

## 2. Models Used
- Detector: YOLO11x (COCO pretrained)
- Tracker: BoT-SORT

## 3. Why This Combination?
BoT-SORT combines Kalman Filter, IOU matching, and ReID features → excellent for sports with occlusion.

## 4. Challenges & Limitations
- Heavy occlusion in crowded scenes
- Similar-looking players
- Fast camera pans

## 5. Possible Improvements
- Fine-tune YOLO on sports dataset
- Stronger ReID model
- Team color clustering
- Speed & trajectory analysis