import os
import pickle
import argparse
import numpy as np
from tqdm import tqdm

def get_mock_embedding(text, dim=768):
    """
    Generate a consistent random embedding for a given text.
    Using a seed based on the text ensures consistency across runs without a real model.
    """
    # Simple hash to seed
    seed = sum([ord(c) for c in text])
    rng = np.random.RandomState(seed)
    return rng.randn(dim).astype(np.float32)

def precompute_embeddings(args):
    """
    Pre-computes embeddings for a set of instructions.
    """
    print(f"Pre-computing embeddings...")
    
    # 1. Gather Instructions
    # In a real scenario, we'd scan the dataset. 
    # For now, we'll use a predefined list or a file.
    if args.instructions_file and os.path.exists(args.instructions_file):
        with open(args.instructions_file, 'r') as f:
            instructions = [line.strip() for line in f if line.strip()]
    else:
        # Default set for testing/mocking
        print("No instruction file provided. Using default test set.")
        instructions = [
            "open the drawer", 
            "pick up the red block", 
            "move the slider left",
            "push the button",
            "close the drawer",
            "place the block in the bin"
        ]
        
    print(f"Found {len(instructions)} unique instructions.")
    
    # 2. Compute Embeddings
    embeddings = {}
    
    # Load existing if updating
    if os.path.exists(args.output) and not args.overwrite:
        print(f"Loading existing embeddings from {args.output}")
        with open(args.output, 'rb') as f:
            embeddings = pickle.load(f)
            
    # Filter out already computed
    to_compute = [i for i in instructions if i not in embeddings]
    print(f"Computing {len(to_compute)} new embeddings...")
    
    if args.mock:
        print("Using MOCK embeddings (random, deterministic).")
    else:
        print("Using REAL embeddings (Placeholder for API).")
        # TODO: Initialize Gemini/Vertex AI client here
        pass

    for instr in tqdm(to_compute):
        if args.mock:
            emb = get_mock_embedding(instr, dim=args.dim)
        else:
            # TODO: Call actual API
            # response = client.embed_text(instr)
            # emb = response.embedding
            # Placeholder for now until API is integrated
            print(f"Warning: Real API not connected. Falling back to mock for '{instr}'")
            emb = get_mock_embedding(instr, dim=args.dim)
            
        embeddings[instr] = emb
        
    # 3. Save
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'wb') as f:
        pickle.dump(embeddings, f)
        
    print(f"Saved {len(embeddings)} embeddings to {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default='training/instruction_embeddings.pkl', help='Output pickle file')
    parser.add_argument('--instructions_file', type=str, default=None, help='Text file with one instruction per line')
    parser.add_argument('--mock', action='store_true', help='Use mock random embeddings instead of API')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing file')
    parser.add_argument('--dim', type=int, default=768, help='Embedding dimension')
    
    args = parser.parse_args()
    precompute_embeddings(args)
