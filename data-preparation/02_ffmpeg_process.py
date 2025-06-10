# 02_ffmpeg_process.py
import os
import json
import subprocess
from tqdm import tqdm
import config
import math

def timestamp_to_seconds(ts: str) -> float:
    """Converts HH:MM:SS.mmm or HH:MM:SS:ms format to total seconds."""
    ts = ts.strip().replace('.', ':')
    parts = ts.split(':')
    
    h, m, s, ms = 0, 0, 0, 0
    
    if len(parts) == 4:
        h, m, s, ms = map(int, parts)
    elif len(parts) == 3:
        m, s, ms = map(int, parts)
    elif len(parts) == 2:
        s, ms = map(int, parts)
    else:
        s = int(parts[0]) if parts[0] else 0

    return float(h) * 3600 + float(m) * 60 + float(s) + float(ms) / 1000.0

def seconds_to_timestamp(s: float) -> str:
    """Converts total seconds to HH:MM:SS.mmm format."""
    h = math.floor(s / 3600)
    s -= h * 3600
    m = math.floor(s / 60)
    s -= m * 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"

def main():
    print("\n--- Stage 2: FFmpeg Video Processing (Scene-Based) ---")
    os.makedirs(config.SEGMENTS_3S_DIR, exist_ok=True)

    annotation_files = sorted([f for f in os.listdir(config.ANNOTATIONS_DIR) if f.endswith('.json')])

    for ann_file in tqdm(annotation_files, desc="Processing Episodes"):
        episode_name = os.path.splitext(ann_file)[0]
        episode_video_path = os.path.join(config.RAW_VIDEO_DIR, f"{episode_name}.mp4")
        if not os.path.exists(episode_video_path):
            print(f"Warning: Video file not found for {episode_name}, skipping.")
            continue

        with open(os.path.join(config.ANNOTATIONS_DIR, ann_file), 'r') as f:
            data = json.load(f)

        for scene in data["scenes"]:
            scene_id = scene["scene_id"]
            
            # Create a dedicated directory for this scene's segments
            scene_segments_dir = os.path.join(config.SEGMENTS_3S_DIR, episode_name, f"scene_{scene_id:03d}")
            os.makedirs(scene_segments_dir, exist_ok=True)

            # Check if segments already exist
            if os.listdir(scene_segments_dir):
                continue
                
            start_sec = timestamp_to_seconds(scene["start_time"])
            end_sec = timestamp_to_seconds(scene["end_time"])
            duration_sec = end_sec - start_sec

            if duration_sec < 2.0:
                print(f"Skipping scene {scene_id} in {episode_name}: duration too short ({duration_sec:.2f}s)")
                continue

            # Crop scene to a multiple of 3 seconds
            target_duration = math.floor(duration_sec / 3.0) * 3.0
            if target_duration == 0:
                continue

            trim_sec = (duration_sec - target_duration) / 2.0
            new_start_sec = start_sec + trim_sec
            
            # Loop through the target duration in 3-second intervals
            num_segments = int(target_duration / 3.0)
            for i in range(num_segments):
                segment_start_sec = new_start_sec + (i * 3.0)
                segment_start_ts = seconds_to_timestamp(segment_start_sec)
                
                output_path = os.path.join(scene_segments_dir, f"segment_{i:03d}.mp4")

                ffmpeg_cmd = [
                    "ffmpeg", "-y",
                    "-ss", segment_start_ts,
                    "-i", episode_video_path,
                    "-t", "3",
                    "-vf", "scale=720:480",
                    "-r", "16",
                    "-frames:v", "48", # Guarantee exactly 48 frames
                    "-c:v", "h264_nvenc", "-preset", "p1", "-cq", "23",
                    "-an", output_path,
                ]

                try:
                    subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                except subprocess.CalledProcessError as e:
                    print(f"FFmpeg error on segment {i} for scene {scene_id} in {episode_name}:")
                    print(f"  Command: {' '.join(e.cmd)}")
                    print(f"  Stderr: {e.stderr}")
                    # Break the inner loop and go to the next scene on error
                    break
                    
    print("Video processing complete.")

if __name__ == "__main__":
    main()