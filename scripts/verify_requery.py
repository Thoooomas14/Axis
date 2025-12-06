import argparse
import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from training.data.rtx_stream_loader import RTXStreamLoader

def visualize_requery(args):
    print(f"Initializing Loader for {args.dataset}...")
    loader = RTXStreamLoader(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        window_size=8,
        batch_size=1
    )
    
    # Initialize TFDS builder
    if loader.builder is None:
        print("Error: Could not initialize dataset builder.")
        return

    # Get one episode directly from the TFDS dataset
    print("Fetching one episode...")
    ds = loader.builder.as_dataset(split='train', shuffle_files=True)
    
    # Take the first one
    episode = next(iter(ds))
    
    print("Processing episode into windows...")
    windows = list(loader._process_episode(episode))
    
    if not windows:
        print("Error: No windows generated (episode too short?).")
        return
        
    print(f"Generated {len(windows)} windows.")
    
    frames = []
    
    # Create a font
    try:
        # Try to load a standard font
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        # Fallback
        font = ImageFont.load_default()

    print("Generating Visualization...")
    for i, window in enumerate(windows):
        # Extract the last image of the window (the "current" observation)
        # Window images shape: (8, 3, 128, 128)
        imgs = window['images'] # Torch tensor or numpy? Loader yields tensors.
        if isinstance(imgs, torch.Tensor):
            imgs = imgs.numpy()
            
        last_img = imgs[-1] # (3, 128, 128)
        
        # CHW -> HWC
        last_img = np.transpose(last_img, (1, 2, 0))
        last_img = (last_img * 255).astype(np.uint8)
        
        pil_img = Image.fromarray(last_img)
        draw = ImageDraw.Draw(pil_img)
        
        # Get Requery Label
        # Shape (1,)
        req = window['requery']
        if isinstance(req, torch.Tensor):
            req = req.item()
        
        # Color code
        color = "green"
        text = "REQUERY: NO"
        if req > 0.5:
            color = "red"
            text = "REQUERY: YES"
        
        # Draw Text
        draw.text((5, 5), f"Step: {i}", fill="white", stroke_width=2, stroke_fill="black", font=font)
        draw.text((5, 105), text, fill=color, stroke_width=2, stroke_fill="black", font=font)
        
        # Add border if requery
        if req > 0.5:
            draw.rectangle([(0,0), (127,127)], outline="red", width=5)
            
        frames.append(pil_img)
        
    # Save GIF
    out_path = "requery_viz.gif"
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=100, # 10fps
        loop=0
    )
    print(f"\n✅ Visualization saved to {out_path}")
    print("Inspect this GIF to see exactly when the model is told to 'Requery'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='fractal20220817_data')
    parser.add_argument('--data_dir', type=str, default=None)
    args = parser.parse_args()
    
    visualize_requery(args)
