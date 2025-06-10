import imageio
from tqdm import tqdm
import os
from os import path as osp
import torch
import torchvision.transforms as TT
import torch.distributed as dist

from ttt.models.configs import VaeModelConfig
from ttt.models.vae.autoencoder import VideoAutoencoderInferenceWrapper
from ttt.infra.parallelisms import init_distributed
from ttt.infra.config_manager import JobConfig
from typing import List

def pad_video(frames: List[torch.Tensor], target_num_frames: int) -> List[torch.Tensor]:
    """Pad video frames to reach target number of frames by repeating the last frame.
    
    Args:
        frames: List of video frames as tensors
        target_num_frames: Target number of frames to pad to
        
    Returns:
        Padded list of frames
    """
    pad_num = target_num_frames - len(frames)
    return frames + [frames[-1]] * pad_num

def crop_video(frames: List[torch.Tensor], target_num_frames: int) -> List[torch.Tensor]:
    """Crop video frames to target number of frames by removing frames from the middle.
    
    Args:
        frames: List of video frames as tensors
        target_num_frames: Target number of frames to crop to
        
    Returns:
        Cropped list of frames
    """
    overflow = len(frames) - target_num_frames
    start = overflow // 2
    return frames[start : start + target_num_frames]

def get_vae(
    vae_weight_path: str,
    fps: int = 16,
    tiling_window_unit: int = 1,
    device: str = "cuda"
) -> VideoAutoencoderInferenceWrapper:
    """Initialize and return a VAE model.
    
    Args:
        vae_weight_path: Path to VAE weights
        fps: Frames per second
        tiling_window_unit: Temporal tiling window unit
        device: Device to load model on
        
    Returns:
        Initialized VAE model
    """
    
    vae = VideoAutoencoderInferenceWrapper(
        vae_weight_path,
        VaeModelConfig.get_encoder_config(temporal_tiling_window=fps*tiling_window_unit),
        VaeModelConfig.get_decoder_config(temporal_tiling_window=2),
    ).to(device).to(torch.bfloat16)
    
    vae.eval()
    print(f"VAE initialized: (enc tiling window = {vae.encoder_temporal_tiling_window}, "
          f"dec tiling window = {vae.decoder_temporal_tiling_window})")
    
    return vae

def precompute_scene_segments(
    videos_dir: str,
    save_dir: str,
    vae: VideoAutoencoderInferenceWrapper,
    target_num_frames: int,
    fps: int,
    batch_size: int = 1
) -> None:
    """Precompute VAE encodings for all videos in a directory."""
    if not osp.exists(save_dir):
        os.makedirs(save_dir)
    
    video_files = sorted([f for f in os.listdir(videos_dir) if f.endswith(".mp4")])
    assert osp.isdir(videos_dir), f"Input directory does not exist: {videos_dir}"
    
    device = vae.device
    dtype = vae.dtype
    batch = []
    batch_save_paths = []

    for i, video in tqdm(enumerate(video_files), total=len(video_files), desc="Processing Segments"):
        video_path = osp.join(videos_dir, video)
        save_path = osp.join(save_dir, video.replace(".mp4", ".pt"))
        
        # Skip if already processed
        if osp.exists(save_path):
            try:
                t = torch.load(save_path, map_location=torch.device('cpu'))
                assert t.shape == (target_num_frames//4+1, 32, 60, 90)
                assert t.dtype == vae.dtype
                assert t[:,:16].min() > -10 and t[:,:16].max() < 10, "mean out of range"
                assert t[:,16:].min() > -40 and t[:,16:].max() < 10, "log var out of range"
                continue
            except Exception as e:
                print(f"Redoing {video} due to error: {e}")
        
        # Process video
        try:
            video_reader = imageio.get_reader(video_path, "ffmpeg")
            video_fps = video_reader.get_meta_data()["fps"]
            frames = [TT.ToTensor()(frame) for frame in video_reader]
            video_reader.close()
        except Exception as e:
            print(f"\n[WARNING] Skipping corrupted video {video_path}. Error: {e}")
            continue

        if int(round(video_fps)) != fps:
            print(f"\n[WARNING] Skipping video with incorrect FPS. Expected {fps}, but got {video_fps:.2f} for {video_path}")
            continue

        # Pad or crop frames to the target length
        if len(frames) < target_num_frames:
            frames = pad_video(frames, target_num_frames)
        elif len(frames) > target_num_frames:
            frames = crop_video(frames, target_num_frames)

        assert len(frames) == target_num_frames, f"Frame count mismatch after padding/cropping for {video_path}"
        
        # Add to batch
        batch.append(torch.stack(frames).to(device).permute(1, 0, 2, 3).unsqueeze(0).to(dtype).contiguous())
        batch_save_paths.append(save_path)
        
        # Process batch if full or last video
        if len(batch) < batch_size and i != len(video_files)-1:
            continue
            
        assert len(batch) <= batch_size and len(batch) == len(batch_save_paths)
        
        # Process batch
        x = torch.cat(batch, dim=0).contiguous()
        x = x * 2.0 - 1.0
        
        expected_shape = (3, target_num_frames, 480, 720)
        assert x.shape[1:] == expected_shape, f"Tensor shape mismatch. Expected {expected_shape}, but got {x.shape[1:]}"
        
        with torch.no_grad():
            encoded_frames = vae.encode_first_stage(x, unregularized=True, multiply_by_scale_factor=False)
            
        # Save encoded frames
        for ef, save_path in zip(encoded_frames, batch_save_paths):
            out = ef.permute((1,0,2,3)).contiguous()
            assert out.shape == (target_num_frames//4+1, 32, 60, 90)
            torch.save(out, save_path)
            
        batch = []
        batch_save_paths = []

        # Clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def main():
    """Main function to precompute video encodings."""
    config = JobConfig()
    config.parse_args()
    init_distributed(config)

    # Get configuration
    episode_dir = config.precomp.episode_dir
    FPS = config.precomp.fps
    video_length = config.precomp.video_length
    tiling_window_unit = config.precomp.vae_tiling_window_unit
    batch_size = config.precomp.batch_size
    vae_weight_path = config.precomp.vae_weight_path
    output_dir = config.precomp.output_dir

    TARGET_NUM_FRAMES = FPS * video_length + 1
    print(f'Precomputing {video_length}s video embeddings\n\tfrom: {episode_dir}\n\tto: {output_dir}.')
    print(f'.mp4 files should have {TARGET_NUM_FRAMES} frames.')

    # Initialize VAE
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    vae = get_vae(
        vae_weight_path=vae_weight_path,
        fps=FPS,
        tiling_window_unit=tiling_window_unit,
        device=f"cuda:{local_rank}"
    )
    assert vae.encoder_temporal_tiling_window == FPS*tiling_window_unit

    # Process episodes
    episodes = sorted([d for d in os.listdir(episode_dir) if osp.isdir(osp.join(episode_dir, d))])
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    episodes_for_this_rank = episodes[rank::world_size]

    for episode in tqdm(episodes_for_this_rank, desc="Processing Episodes"):
        episode_path = osp.join(episode_dir, episode)
        scenes = sorted([s for s in os.listdir(episode_path) if osp.isdir(osp.join(episode_path, s))])
        
        for scene in scenes:
            scene_path = osp.join(episode_path, scene)
            output_scene_path = osp.join(output_dir, episode, scene)

            precompute_scene_segments(
                videos_dir=scene_path,
                save_dir=output_scene_path,
                vae=vae,
                target_num_frames=TARGET_NUM_FRAMES,
                fps=FPS,
                batch_size=batch_size
            )
        print(f'Done processing episode {episode}!')
    
    dist.barrier()

if __name__ == "__main__":
    main()