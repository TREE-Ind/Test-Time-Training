# 04_assemble_metadata.py
import os
import json
from tqdm import tqdm
import config

def main():
    """
    Assembles the final training metadata JSONL file by combining the
    concatenated video latents with their corresponding text embeddings.
    """
    print("\n--- Stage 4: Assembling Final Metadata JSONL ---")

    os.makedirs(config.METADATA_DIR, exist_ok=True)
    
    input_jsonl_path = os.path.join(config.PRECOMP_TEXT_INPUT_DIR, "precomp_text_input.jsonl")
    output_jsonl_path = os.path.join(config.METADATA_DIR, "final_training_data.jsonl")

    # The output from precompute_text.py saves embeddings in a nested structure.
    # We need to construct the path to each text embedding.
    # The saved filenames are hashes of the input text. We don't have an easy
    # way to map them back without reading the log output of precompute_text.py,
    # which is fragile.
    #
    # A more robust approach, given the project structure, is to assume that
    # the text precomputation script saves embeddings in a way that can be
    # deterministically retrieved. The official script uses a hash of the text.
    # Let's see how precompute_text.py works.
    # For now, let's create a placeholder structure. The user may need to adjust
    # based on the exact output of their precompute_text.py script.

    with open(input_jsonl_path, 'r') as in_f, open(output_jsonl_path, 'w') as out_f:
        for line in tqdm(in_f, desc="Assembling Metadata"):
            data = json.loads(line)
            
            video_latent_path = data["video_latent_path"]

            if not os.path.exists(video_latent_path):
                print(f"Warning: Video latent not found, skipping: {video_latent_path}")
                continue

            episode_name = data["episode_name"]
            scene_id_str = data["scene_id_str"]

            num_segments = len(data["scene_start"])
            
            # This logic now correctly constructs the path to the text embeddings
            # based on the output structure of the modified precompute_text.py script.
            # It expects a structure like:
            # .../text_embeddings/<token_mode_dir>/<episode_name>/<scene_id>/segment_000i_txt_emb.pt
            text_chunk_emb_paths = []
            for i in range(num_segments):
                # We are linking to the embeddings generated with the default (empty) token_mode.
                # The training script can be configured to use other versions (e.g., with scene tokens).
                text_emb_path = os.path.join(
                    config.TEXT_EMBEDDINGS_DIR, 
                    "tom-and-jerry-3s-120-", # This is the sub-directory for the default token_mode
                    episode_name,
                    scene_id_str,
                    f"segment_{i:04d}_txt_emb.pt"
                )
                text_chunk_emb_paths.append(os.path.relpath(text_emb_path, config.DATASET_OUTPUT_PATH))


            final_obj = {
                "vid_emb": os.path.relpath(video_latent_path, config.DATASET_OUTPUT_PATH),
                "text_chunk_emb": text_chunk_emb_paths
            }
            
            out_f.write(json.dumps(final_obj) + '\n')

    print(f"Final metadata file created at {output_jsonl_path}")
    print("\nMetadata assembly complete. The 'final_training_data.jsonl' is ready for training.")


if __name__ == "__main__":
    main()