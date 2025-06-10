# 03a_concatenate_latents.py
import os
import torch
from tqdm import tqdm
import config

def main():
    """
    Concatenates individual 3-second video segment latents (.pt files)
    into a single latent file for each scene.
    """
    print("\n--- Stage 3a: Concatenating Scene Latents ---")
    
    # Ensure the final output directory exists
    os.makedirs(config.VIDEO_LATENTS_CONCAT_DIR, exist_ok=True)
    
    # Iterate through episodes
    for episode_name in tqdm(sorted(os.listdir(config.VIDEO_LATENTS_DIR)), desc="Concatenating Episodes"):
        episode_path = os.path.join(config.VIDEO_LATENTS_DIR, episode_name)
        if not os.path.isdir(episode_path):
            continue

        # Iterate through scenes within an episode
        for scene_name in sorted(os.listdir(episode_path)):
            scene_path = os.path.join(episode_path, scene_name)
            if not os.path.isdir(scene_path):
                continue

            # Define the path for the final concatenated latent file
            output_latent_path = os.path.join(config.VIDEO_LATENTS_CONCAT_DIR, f"{episode_name}_{scene_name}.pt")

            # Skip if the final file already exists
            if os.path.exists(output_latent_path):
                continue
            
            segment_files = sorted([f for f in os.listdir(scene_path) if f.endswith('.pt')])
            if not segment_files:
                continue

            # Load and concatenate all segment tensors
            all_segments = []
            for segment_file in segment_files:
                latent_path = os.path.join(scene_path, segment_file)
                latent_tensor = torch.load(latent_path, map_location='cpu')
                all_segments.append(latent_tensor)
            
            # Concatenate along the temporal dimension (dim=2 for B, C, T, H, W)
            # Assuming latent shape is (C, T, H, W), we add a batch dim and then concat on T.
            # Then remove the batch dim.
            # Let's check a shape first.
            if len(all_segments) > 0:
                # Assuming shape is [C, T_seg, H, W]. We want [C, T_full, H, W]
                # The temporal dimension for VAE latents from this project is dim=1
                concatenated_latents = torch.cat(all_segments, dim=1)
                
                # Save the final tensor
                torch.save(concatenated_latents, output_latent_path)

    print("Latent concatenation complete.")

if __name__ == "__main__":
    main() 