import json
import os
import numpy as np
import torch
import argparse
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import T5Tokenizer, T5EncoderModel

# Constants
SCENE_END_TOKEN = "<end_scene>"
SCENE_START_TOKEN = "<start_scene>"

def expand_dims_like(x, y):
    """Expand dimensions of x to match y's dimensionality."""
    while x.dim() != y.dim():
        x = x.unsqueeze(-1)
    return x

class TextDataset(Dataset):
    """Dataset class for handling JSONL data with scene tokens."""
    def __init__(self, json_file, token_mode=""):
        self.entries = []
        assert token_mode in ["start", "end", "both", ""]
        
        with open(json_file, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                scene_data = json.loads(line)
                
                episode_name = scene_data["episode_name"]
                scene_id_str = scene_data["scene_id_str"]
                
                texts = []
                i = 0
                while f"text_{i}" in scene_data:
                    texts.append(scene_data[f"text_{i}"])
                    i += 1
                
                if not texts:
                    continue

                scene_starts = scene_data['scene_start']
                scene_ends = scene_data['scene_end']
                
                for i, text in enumerate(texts):
                    entry = {}
                    text_to_process = text
                    
                    is_scene_start = scene_starts[i]
                    is_scene_end = scene_ends[i]
                    
                    if token_mode == "end":
                        if not is_scene_end: continue
                        text_to_process = text + SCENE_END_TOKEN
                    elif token_mode == "start":
                        if not is_scene_start: continue
                        text_to_process = SCENE_START_TOKEN + text
                    elif token_mode == "both":
                        if not (is_scene_start and is_scene_end): continue
                        text_to_process = SCENE_START_TOKEN + text + SCENE_END_TOKEN
                    
                    entry['text'] = text_to_process
                    # This path is not a real file, it's just used to construct the output path.
                    entry['path'] = os.path.join(episode_name, scene_id_str, f"segment_{i:04d}.mp4")
                    self.entries.append(entry)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        return self.entries[idx]

def collate_fn(batch):
    """Simple collate function that returns the batch unchanged."""
    return batch

def extract_batch_embeddings(batch_texts, model, tokenizer, device, max_length, padding="max_length"):
    """Extract embeddings for a batch of texts using the T5 model."""
    inputs = tokenizer(
        batch_texts,
        truncation=True,
        max_length=max_length,
        return_length=True,
        return_overflowing_tokens=True,
        padding=padding,
        return_tensors="pt",
    ).to(device)

    tokens = inputs["input_ids"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=tokens)

    embeddings = outputs.last_hidden_state
    return embeddings.detach()

def process_jsonl(model, tokenizer, input_jsonl_file, output_path, max_length, token_mode=None, batch_size=16):
    """Process JSONL file and extract text embeddings."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    num_added_toks = tokenizer.add_special_tokens({'additional_special_tokens': [SCENE_END_TOKEN, SCENE_START_TOKEN]})
    model.resize_token_embeddings(len(tokenizer))
    
    model = torch.nn.DataParallel(model)
    model = model.to(device).to(torch.bfloat16)
    model.eval()

    dataset = TextDataset(input_jsonl_file, token_mode=token_mode)
    print(f'Loaded {len(dataset)} data')
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn, drop_last=False
    )

    for batch in tqdm(dataloader, desc="Processing batches"):
        batch_texts = [entry["text"] for entry in batch]
        batch_paths = [entry["path"] for entry in batch]

        embeddings = extract_batch_embeddings(batch_texts, model, tokenizer, device, max_length)

        for i, (embedding, path) in enumerate(zip(embeddings, batch_paths)):
            sub_dir = os.path.dirname(path)
            video_filename = os.path.basename(path).replace(".mp4", "_txt_emb.pt")
            emb_path = os.path.join(output_path, sub_dir, video_filename)

            os.makedirs(os.path.dirname(emb_path), exist_ok=True)
            torch.save(embedding, emb_path)
            assert embedding.shape[0] == max_length, f"Embedding shape mismatch: {embedding.shape[0]}"
    print(f"Processing complete. Text embeddings saved to {output_path}.")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Process text embeddings for video data')
    parser.add_argument('--video_length', type=int, default=3, help='Length of videos in seconds')
    parser.add_argument('--max_length', type=int, default=493, help='Maximum sequence length')
    parser.add_argument('--checkpoint_dir', type=str, required=True, help='Path to T5 checkpoint directory')
    parser.add_argument('--input_jsonl_file', type=str, required=True, help='Name of input JSONL file')
    parser.add_argument('--output_path', type=str, required=True, help='Base path for output embeddings')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for processing')
    return parser.parse_args()

def main():
    args = parse_args()

    token_modes = ["", "both", "start", "end"]
    print(f'Processing {args.input_jsonl_file}(video length {args.video_length}s) with max_length={args.max_length}')

    tokenizer = T5Tokenizer.from_pretrained(args.checkpoint_dir)
    model = T5EncoderModel.from_pretrained(args.checkpoint_dir)
    print(f"Loading model and tokenizer from checkpoint: {args.checkpoint_dir}")

    for token_mode in token_modes:
        print(f'Processing token_mode={token_mode}')
        output_path = os.path.join(args.output_path, f"tom-and-jerry-{args.video_length}s-{args.max_length}{'-'+token_mode}")
        process_jsonl(model, tokenizer, args.input_jsonl_file, output_path, 
                    args.max_length, batch_size=args.batch_size, token_mode=token_mode)

if __name__ == "__main__":
    main()