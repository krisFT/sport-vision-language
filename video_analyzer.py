import cv2
import os
import json
import numpy as np
from Image_Analyzer_AI import ImageAnalyzer

# Helper to convert numpy types for JSON serialization
def to_serializable(val):
    if isinstance(val, (np.integer,)):
        return int(val)
    elif isinstance(val, (np.floating,)):
        return float(val)
    elif isinstance(val, (np.ndarray,)):
        return val.tolist()
    return val

# Path to your video file
video_path = "test.mp4"
output_dir = "video_frames_output"
os.makedirs(output_dir, exist_ok=True)

# Initialize your analyzer
analyzer = ImageAnalyzer()

# Open the video
cap = cv2.VideoCapture(video_path)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)

frame_idx = 0
save_every_n = 100  # Analyze every 100th frame (changed from 30)

all_results = []
all_conversations = []

# For scene memory: keep previous dialog (as a summary string)
previous_dialog = None

def parse_dialog(scene_analysis):
    """
    Cleans and parses the LLM output.
    Handles cases where the output is a markdown JSON block.
    """
    try:
        # Clean the string: remove markdown fences and strip whitespace
        clean_str = scene_analysis.strip()
        if clean_str.startswith("```json"):
            clean_str = clean_str[7:]
        if clean_str.startswith("```"):
            clean_str = clean_str[3:]
        if clean_str.endswith("```"):
            clean_str = clean_str[:-3]
        
        dialog = json.loads(clean_str.strip())
        
        if isinstance(dialog, list) and all(isinstance(turn, dict) and 'speaker' in turn and 'utterance' in turn for turn in dialog):
            return dialog
    except Exception as e:
        print(f"Error parsing dialog, falling back. Error: {e}")
        pass
    
    # Fallback for non-JSON or malformed output
    return [{"speaker": "Alex", "utterance": scene_analysis.strip()}]

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx % save_every_n == 0:
        frame_filename = os.path.join(output_dir, f"frame_{frame_idx:05d}.jpg")
        cv2.imwrite(frame_filename, frame)
        print(f"Processing frame {frame_idx}/{frame_count}")

        # Run analysis, passing previous dialog for context
        results = analyzer.analyze_image(frame_filename, previous_dialog=previous_dialog)

        # Check for meaningful info
        has_faces = results.get("object_detection", {}).get("people_count", 0) > 0
        has_teams = bool(results.get("ner_entities", {}).get("TEAM"))
        has_ocr = any(t.get("confidence", 0) > 0.5 for t in results.get("ocr_texts", []))

        if has_faces or has_teams or has_ocr:
            # Save enhanced visualization (already saved by analyze_image)
            enhanced_name = os.path.join(output_dir, f"enhanced_{frame_idx:05d}.jpg")
            if os.path.exists("output_enhanced.jpg"):
                os.rename("output_enhanced.jpg", enhanced_name)

            # Parse dialog from LLM output
            scene_analysis = results.get("scene_analysis", "")
            dialog = parse_dialog(scene_analysis)

            # Save LLM analysis for this frame
            frame_result = {
                "frame_index": frame_idx,
                "dialog": dialog,  # Store as structured dialog
                "object_detection": results.get("object_detection", {}),
                "ocr_texts": results.get("ocr_texts", []),
                "ner_entities": results.get("ner_entities", {}),
                # Optionally, store previous dialog for frontend use
                "previous_dialog": previous_dialog
            }
            all_results.append(frame_result)
            # Save per-frame JSON, ensuring serializability
            with open(os.path.join(output_dir, f"analysis_{frame_idx:05d}.json"), "w") as f:
                json.dump(frame_result, f, indent=2, default=to_serializable)
            # Collect conversation for all_conversations.json
            conversation = {
                "frame_index": frame_idx,
                "dialog": dialog
            }
            all_conversations.append(conversation)
            # Update previous_dialog for next frame (as a summary string)
            if dialog:
                # Use the last utterance as the summary for the next frame
                previous_dialog = dialog[-1]["utterance"]
            else:
                previous_dialog = None
        # else: skip this frame

    frame_idx += 1

cap.release()

# Save all results to a single JSON file
with open(os.path.join(output_dir, "all_analysis_results.json"), "w") as f:
    json.dump(all_results, f, indent=2, default=to_serializable)

# Save all conversations to a single JSON file
with open(os.path.join(output_dir, "all_conversations.json"), "w") as f:
    json.dump(all_conversations, f, indent=2, default=to_serializable)

print("Video analysis complete.") 