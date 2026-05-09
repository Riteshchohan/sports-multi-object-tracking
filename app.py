import gradio as gr
import cv2
from ultralytics import YOLO
import supervision as sv
import tempfile
import os
import traceback

# Load model
model = YOLO("yolo11s.pt")

def process_video(video_file, confidence: float = 0.4):
    if video_file is None:
        return None, "❌ Please upload a video file"

    try:
        # Handle both cases: file object or file path (string)
        if isinstance(video_file, str):
            input_path = video_file
        else:
            # It's a file object
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_in:
                tmp_in.write(video_file.read())
                input_path = tmp_in.name

        output_path = input_path.replace(".mp4", "_tracked.mp4")

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            return None, "❌ Could not open the video file. Please try another video."

        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        box_annotator = sv.BoxAnnotator(thickness=3)
        label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=1.0)
        trace_annotator = sv.TraceAnnotator(thickness=2, trace_length=50)

        frame_count = 0
        max_frames = 800   # Limit for free CPU

        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            try:
                results = model.track(
                    frame, 
                    persist=True, 
                    conf=confidence,
                    classes=[0], 
                    verbose=False
                )[0]

                detections = sv.Detections.from_ultralytics(results)

                annotated_frame = frame.copy()
                annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)

                if detections.tracker_id is not None:
                    labels = [f"ID: {int(tid)}" for tid in detections.tracker_id]
                    annotated_frame = label_annotator.annotate(
                        scene=annotated_frame, 
                        detections=detections, 
                        labels=labels
                    )

                annotated_frame = trace_annotator.annotate(scene=annotated_frame, detections=detections)
                out.write(annotated_frame)
                frame_count += 1

            except Exception:
                out.write(frame)   # Write original frame if tracking fails
                frame_count += 1

        cap.release()
        out.release()

        return output_path, f"✅ Success! Processed {frame_count} frames with persistent IDs."

    except Exception as e:
        print(traceback.format_exc())
        return None, f"❌ Error: {str(e)}"

# ====================== UI ======================
with gr.Blocks(title="Sports Multi-Object Tracking", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# ⚽ Multi-Object Detection & Persistent ID Tracking\n**YOLO11s + BoT-SORT**")

    with gr.Row():
        with gr.Column():
            input_video = gr.Video(label="Upload Sports Video (Short clips work best)")
            confidence = gr.Slider(0.3, 0.8, value=0.4, step=0.05, label="Confidence Threshold")
            process_button = gr.Button("🚀 Start Tracking", variant="primary", size="large")

        with gr.Column():
            output_video = gr.Video(label="Output: Tracked Video with IDs")

    status = gr.Textbox(label="Status")

    process_button.click(
        fn=process_video,
        inputs=[input_video, confidence],
        outputs=[output_video, status]
    )

    gr.Markdown("**Tip**: Use short sports videos (15–40 seconds) for best results.")

if __name__ == "__main__":
    demo.launch()