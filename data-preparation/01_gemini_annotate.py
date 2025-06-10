# 01_gemini_annotate.py
import os
import json
import pathlib
from google import genai
from google.genai import types
from tqdm import tqdm
import time
import config
import datetime
from pydantic import BaseModel
from typing import List

# --- Pydantic Schemas for Structured Output ---
class Segment(BaseModel):
    segment_start_time: str
    segment_end_time: str
    annotation_text: str

class Scene(BaseModel):
    scene_id: int
    start_time: str
    end_time: str
    segments: List[Segment]

class VideoAnnotation(BaseModel):
    scenes: List[Scene]

# --- Gemini API Prompt for One-Pass Annotation ---
ONE_PASS_ANNOTATION_PROMPT = """
Your task is to analyze a video and generate a detailed JSON annotation for training a video model.

First, identify all distinct scenes. A scene is defined by a significant change in location, background, time of day, or narrative focus.

Second, for each scene you identify, you must break it down into consecutive 3-second clips.

Third, for each 3-second clip, provide a detailed, 2-3 sentence description of the primary action, character interactions, and camera movement within that specific clip.

Your output **must** be a single, valid JSON object that follows this exact schema, with the `scenes` array containing all the scenes from the video:

{
  "scenes": [
    {
      "scene_id": 1,
      "start_time": "00:00:00.000",
      "end_time": "00:00:25.500",
      "segments": [
        {
          "segment_start_time": "00:00:00.000",
          "segment_end_time": "00:00:03.000",
          "annotation_text": "A detailed 2-3 sentence description of the action in this 3-second clip."
        },
        {
          "segment_start_time": "00:00:03.000",
          "segment_end_time": "00:00:06.000",
          "annotation_text": "The description for the next 3-second clip."
        }
      ]
    }
  ]
}

- Ensure all timestamps are in the precise 'HH:MM:SS.ms' format.
- The `segments` array for each scene must contain consecutive, non-overlapping 3-second clips that cover the entire duration of the scene.
"""

def main():
    """Analyzes videos to generate a complete, granular annotation in a single pass."""
    print("--- Stage 1: Gemini Annotation (One-Pass Granular) ---")
    
    if config.GEMINI_API_KEY == "YOUR_API_KEY_HERE":
        print("ERROR: Please set your Gemini API key in config.py")
        return

    try:
        client = genai.Client(api_key=config.GEMINI_API_KEY, http_options=types.HttpOptions(timeout=60000))
    except Exception as e:
        print(f"Error initializing Gemini client: {e}")
        return

    os.makedirs(config.ANNOTATIONS_DIR, exist_ok=True)
    
    model_name = 'gemini-2.0-flash'
    
    video_files = sorted([f for f in os.listdir(config.RAW_VIDEO_DIR) if f.lower().endswith(('.mp4', '.mov', '.avi'))])
    
    for video_file in tqdm(video_files, desc="Processing Episodes"):
        episode_name = pathlib.Path(video_file).stem
        video_path = os.path.join(config.RAW_VIDEO_DIR, video_file)
        output_json_path = os.path.join(config.ANNOTATIONS_DIR, f"{episode_name}.json")

        if os.path.exists(output_json_path):
            tqdm.write(f"Skipping {video_file}, annotation already exists.")
            continue

        video_file_obj = None
        try:
            tqdm.write(f"\nUploading {video_file} to Gemini... This may take a moment.")
            video_file_obj = client.files.upload(file=video_path)
            tqdm.write(f"Successfully uploaded {video_file}. Waiting for processing...")

            while video_file_obj.state.name == "PROCESSING":
                time.sleep(10)
                video_file_obj = client.files.get(name=video_file_obj.name)

            if video_file_obj.state.name != "ACTIVE":
                raise Exception(f"File processing failed. State: {video_file_obj.state.name}")

            tqdm.write("File is active. Generating full annotation... (This can take several minutes)")
            
            # Single API call to get the entire structured annotation
            response = client.models.generate_content(
                model=model_name,
                contents=[ONE_PASS_ANNOTATION_PROMPT, video_file_obj],
                config={
                    "response_mime_type": "application/json",
                    "response_schema": VideoAnnotation,
                },
                #request_options={'timeout': 1200} # Increased timeout for the complex single-pass task
            )
            
            # The structured response is guaranteed to be valid JSON.
            scene_data = json.loads(response.text)

            if "scenes" not in scene_data or not isinstance(scene_data["scenes"], list):
                raise ValueError("Invalid JSON structure received despite schema.")

            # Save the final structured data
            with open(output_json_path, 'w') as f:
                json.dump(scene_data, f, indent=2)

        except Exception as e:
            tqdm.write(f"Error during processing for {video_file}: {e}")
            continue # Continue to the next file
        finally:
            # Always attempt to clean up the file
            if video_file_obj:
                tqdm.write(f"Cleaning up file: {video_file_obj.name}")
                client.files.delete(name=video_file_obj.name)
        
        tqdm.write(f"Finished processing and saved annotations to {output_json_path}")

if __name__ == "__main__":
    main()