import argparse
import pickle
import os
import sys
import torch
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from training.data.rtx_loader import RTXDataset

def get_gemini_embedding(text):
    """
    Mock function to get embedding from Gemini.
    Replace this with actual API call.
    Returns: (768,) vector
    """
    # In reality: 
    # response = genai.embed_content(model="models/embedding-001", content=text)
    # return response['embedding']
    return torch.randn(768).numpy()

def main():
    parser = argparse.ArgumentParser(description="Pre-compute Gemini embeddings for RT-X dataset instructions.")
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='fractal20220817_data', help='Name of the TFDS dataset')
    parser.add_argument('--output', type=str, default='instruction_embeddings.pkl', help='Output pickle file')
    args = parser.parse_args()

    print(f"Scanning dataset {args.dataset} for unique instructions...")
    
    # Initialize dataset (batch size 1 to iterate easily)
    # Note: This requires tensorflow_datasets to be installed and working
    try:
        dataset = RTXDataset(dataset_name=args.dataset, split='train', batch_size=1)
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    unique_instructions = set()
    
    # Iterate through a subset or full dataset to find unique instructions
    # For large datasets, we might want to limit this or use a dedicated metadata file if available.
    # Here we scan the first N steps.
    max_steps = 1000 
    print(f"Scanning first {max_steps} steps...")
    
    for i, (_, _, lang_batch) in tqdm(enumerate(dataset), total=max_steps):
        if i >= max_steps:
            break
        
        # lang_batch is a list/array of bytes
        for lang in lang_batch:
            if isinstance(lang, bytes):
                lang = lang.decode('utf-8')
            unique_instructions.add(lang)

    print(f"Found {len(unique_instructions)} unique instructions.")
    
    # Compute embeddings
    embeddings = {}
    print("Computing embeddings...")
    for instr in tqdm(unique_instructions):
        if not instr:
            continue
        embeddings[instr] = get_gemini_embedding(instr)

    # Save
    with open(args.output, 'wb') as f:
        pickle.dump(embeddings, f)
    
    print(f"Saved embeddings to {args.output}")

if __name__ == "__main__":
    main()
