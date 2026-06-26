import os

import torch
from torchvision.io import read_image

from common import TASK_SET
from common.vision_encoder import PretrainedEncoder

RECOMPUTE = True  # Set to True to recompute features
FILEDIR = "<path>/<to>/<dataset>"

# Load encoder
encoder = PretrainedEncoder()

for task in TASK_SET['soup']:

    # Check whether features have already been computed
    td_path = f"FILEDIR/{task}.pt"
    if os.path.exists(td_path):
        td = torch.load(td_path, weights_only=False)
        if 'feat' in td and not RECOMPUTE:
            print(f"Features already computed for task {task}. Skipping.")
            continue

    # Load image data
    print('Encoding data for task:', task)

    i = 0
    fp = lambda i: f"{FILEDIR}/{task}-{i}.png"
    features = []

    while os.path.exists(fp(i)):
        frames = read_image(fp(i))  # (3, 224, 224*B)
        num_frames = frames.shape[-1] // 224  # Number of images in batch
        frames = frames.view(3, 224, num_frames, 224)  # Reshape to (3, 224, B, 224)
        frames = frames.permute(2, 0, 1, 3)  # Reshape to (B, 3, 224, 224)
        
        # Encode frames in smaller batches
        batch_size = 256
        frame_idx = 0
        while frame_idx < num_frames:
            # Extract batch of frames
            end_idx = frame_idx + batch_size
            if end_idx > num_frames:
                end_idx = num_frames
            batch_frames = frames[frame_idx:end_idx]
            out = encoder(batch_frames)
            features.append(out.cpu())
            print(f'Processed {end_idx}/{num_frames} frames in chunk {i+1} for task {task}, feature dim: {int(out.shape[-1])}')
            frame_idx += batch_size

        i += 1

    if len(features) == 0:
        print(f"No data found for task {task}. Skipping.")
        continue

    # Save features
    features = torch.cat(features, dim=0)  # Concatenate all features
    print(features.shape, features.dtype, features.device)
    features = features[:td['obs'].shape[0]]  # Match number of observations
    print('Final feature shape:', features.shape)
    td['feat'] = features
    torch.save(td, f"{FILEDIR}/{task}.pt")
