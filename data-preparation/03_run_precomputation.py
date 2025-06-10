# 03_run_precomputation.py
import os
import json
import subprocess
import glob
from tqdm import tqdm
import config
import math

def create_text_precomp_input_jsonl():
    """
    Creates the JSONL file needed by precompute_text.py, where each line
    represents a full scene with all its text annotations.
    """
    print("Creating JSONL for text precomputation...")
    output_jsonl_path = os.path.join(config.PRECOMP_TEXT_INPUT_DIR, "precomp_text_input.jsonl")
    os.makedirs(config.PRECOMP_TEXT_INPUT_DIR, exist_ok=True)

    with open(output_jsonl_path, 'w') as out_f:
        for ann_file in tqdm(sorted(os.listdir(config.ANNOTATIONS_DIR)), desc="Processing Annotations"):
            if not ann_file.endswith('.json'):
                continue
            
            episode_name = os.path.splitext(ann_file)[0]
            with open(os.path.join(config.ANNOTATIONS_DIR, ann_file), 'r') as f:
                data = json.load(f)

            for scene in data["scenes"]:
                scene_id = scene["scene_id"]
                
                # Re-calculate the number of segments after cropping
                start_sec = timestamp_to_seconds(scene["start_time"])
                end_sec = timestamp_to_seconds(scene["end_time"])
                duration_sec = end_sec - start_sec
                if duration_sec < 2.0:
                    continue
                target_duration = math.floor(duration_sec / 3.0) * 3.0
                if target_duration == 0:
                    continue
                num_segments = int(target_duration / 3.0)

                # Ensure we don't use more annotations than segments we created
                annotations = [seg["annotation_text"] for seg in scene["segments"]][:num_segments]
                if not annotations:
                    continue

                # Build the JSON object for this line
                json_obj = {
                    "video_latent_path": os.path.join(config.VIDEO_LATENTS_CONCAT_DIR, f"{episode_name}_scene_{scene_id:03d}.pt"),
                    "scene_start": [True] + [False] * (num_segments - 1),
                    "scene_end": [False] * (num_segments - 1) + [True],
                    "episode_name": episode_name,
                    "scene_id_str": f"scene_{scene_id:03d}"
                }
                
                for i, text in enumerate(annotations):
                    json_obj[f"text_{i}"] = text
                
                # Write the JSON object as a line in the output file
                out_f.write(json.dumps(json_obj) + '\n')
                
    print(f"Text precomputation JSONL created at {output_jsonl_path}")
    return output_jsonl_path

# Helper function from 02_ffmpeg_process.py to avoid code duplication
def timestamp_to_seconds(ts: str) -> float:
    ts = ts.strip().replace('.', ':')
    parts = ts.split(':')
    h, m, s, ms = 0, 0, 0, 0
    if len(parts) == 4: h, m, s, ms = map(int, parts)
    elif len(parts) == 3: m, s, ms = map(int, parts)
    elif len(parts) == 2: s, ms = map(int, parts)
    else: s = int(parts[0]) if parts[0] else 0
    return float(h) * 3600 + float(m) * 60 + float(s) + float(ms) / 1000.0

def main():
    """Runs the official precomputation scripts for video."""
    print("\n--- Stage 3: Official Precomputation ---")
    os.makedirs(config.VIDEO_LATENTS_DIR, exist_ok=True)
    
    # --- Part 1: Video Latent Generation ---
    print("\nRunning precompute_video.py to generate individual segment latents...")
    
    # Updated to use torchrun for multi-GPU execution on an 8-GPU node.
    # The batch size is per-GPU. A total of 8 * 8 = 64 videos will be processed in parallel.
    # Adjust batch_size based on your GPU VRAM (4-8 for 40GB A100s, 16-32 for 80GB A100s).
    video_command = [
        'torchrun', 
        '--nproc_per_node=8',
        config.PRECOMP_VIDEO_SCRIPT,
        '--precomp.episode_dir', config.SEGMENTS_3S_DIR,
        '--precomp.output_dir', config.VIDEO_LATENTS_DIR,
        '--precomp.vae_weight_path', config.COGVIDEOX_VAE_PATH,
        '--precomp.video_length', '3',
        '--precomp.fps', str(config.FPS),
        '--precomp.batch_size', '8'  # This is a per-GPU batch size
    ]
    
    # torchrun handles distributed environment variables automatically.
    # We just need to set the PYTHONPATH.
    env = os.environ.copy()
    env['PYTHONPATH'] = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    try:
        subprocess.run(video_command, check=True, env=env)
        print("Individual video latent generation complete.")
    except subprocess.CalledProcessError as e:
        print(f"Error running precompute_video.py: {e}")
        return

    # --- Part 2: Concatenate Video Latents ---
    print("\nRunning 03a_concatenate_latents.py to combine scene latents...")
    try:
        # This is a simple script, can be run directly without special env
        subprocess.run(['python', 'data-preparation/03a_concatenate_latents.py'], check=True)
        print("Video latent concatenation complete.")
    except subprocess.CalledProcessError as e:
        print(f"Error running 03a_concatenate_latents.py: {e}")
        return

    # --- Part 3: Text Embedding Generation ---
    print("\nCreating input file for text precomputation...")
    text_input_jsonl = create_text_precomp_input_jsonl()

    print("\nRunning precompute_text.py...")
    text_command = [
        'python', config.PRECOMP_TEXT_SCRIPT,
        '--input_jsonl_file', text_input_jsonl,
        '--output_path', config.TEXT_EMBEDDINGS_DIR,
        '--checkpoint_dir', config.T5_MODEL_PATH,
        '--max_length', '120', # Using a more standard max_length
        '--batch_size', '16'
    ]
    try:
        subprocess.run(text_command, check=True, env=env)
        print("Text embedding precomputation complete.")
    except subprocess.CalledProcessError as e:
        print(f"Error running precompute_text.py: {e}")

if __name__ == "__main__":
    main()