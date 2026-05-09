"""
Multi-Object Detection and Persistent ID Tracking for Sports Footage
Assignment Solution
"""

import cv2
import time
import argparse
from pathlib import Path
from ultralytics import YOLO
import supervision as sv
from tqdm import tqdm

# ========================= CONFIGURATION =========================
VIDEO_PATH = "input_video.mp4"
OUTPUT_PATH = "annotated_output.mp4"

# Model & Tracker Settings
MODEL_NAME = "yolo11x.pt"        # Options: yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11x.pt
TRACKER = "botsort.yaml"         # "botsort.yaml" is better for occlusion
CONF_THRESHOLD = 0.35
IOU_THRESHOLD = 0.5
CLASSES = [0]                    # 0 = person

# Visualization
SHOW_TRAILS = True
TRAIL_LENGTH = 60
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Multi-object tracking for sports footage")
    parser.add_argument("--source", default=VIDEO_PATH, help="Path to input video file")
    parser.add_argument("--output", default=OUTPUT_PATH, help="Path to output annotated video file")
    args = parser.parse_args()

    print("🚀 Starting Multi-Object Tracking Pipeline...")

    # Load Model
    model = YOLO(MODEL_NAME)
    print(f"✅ Loaded {MODEL_NAME}")

    # Load Video
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print("❌ Error: Could not open video.")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Video Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    # Supervision Annotators
    box_annotator = sv.BoxAnnotator(thickness=3)
    label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=1.0, text_color=sv.Color.WHITE)
    trace_annotator = sv.TraceAnnotator(thickness=2, trace_length=TRAIL_LENGTH) if SHOW_TRAILS else None

    frame_count = 0
    start_time = time.time()

    print(f"📹 Processing video: {Path(args.source).name} | {total_frames} frames")

    with tqdm(total=total_frames, desc="Processing Frames") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Run Detection + Tracking
            results = model.track(
                frame,
                persist=True,
                tracker=TRACKER,
                conf=CONF_THRESHOLD,
                iou=IOU_THRESHOLD,
                classes=CLASSES,
                verbose=False
            )[0]

            # Convert to Supervision Detections
            detections = sv.Detections.from_ultralytics(results)

            # Annotate Frame
            annotated_frame = frame.copy()
            annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)

            # Create labels with ID
            labels = []
            tracker_ids = detections.tracker_id
            if tracker_ids is None:
                labels = ["ID: ?"] * len(detections)
            else:
                for tracker_id in tracker_ids:
                    if tracker_id is None:
                        labels.append("ID: ?")
                    else:
                        labels.append(f"ID: {int(tracker_id)}")

            annotated_frame = label_annotator.annotate(
                scene=annotated_frame,
                detections=detections,
                labels=labels
            )

            # TraceAnnotator requires tracker_id; skip when not available.
            if SHOW_TRAILS and trace_annotator and detections.tracker_id is not None:
                annotated_frame = trace_annotator.annotate(scene=annotated_frame, detections=detections)

            # Write frame
            out.write(annotated_frame)

            frame_count += 1
            pbar.update(1)

    # Cleanup
    cap.release()
    out.release()

    end_time = time.time()
    print(f"\n✅ Processing Complete!")
    print(f"📁 Output saved: {args.output}")
    print(f"⏱️  Time taken: {end_time - start_time:.2f} seconds")
    print(f"📊 Average FPS: {frame_count / (end_time - start_time):.2f}")


if __name__ == "__main__":
    main()