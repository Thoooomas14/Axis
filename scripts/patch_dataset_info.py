import json
import os

def patch_dataset_info():
    info_path = 'data/content/axis_data/fractal20220817_data/0.1.0/dataset_info.json'
    
    if not os.path.exists(info_path):
        print(f"Error: {info_path} not found.")
        return

    with open(info_path, 'r') as f:
        data = json.load(f)

    # We have shards 00000 to 00099, so 100 shards.
    # We need to truncate 'shardLengths' in the 'train' split.
    
    splits = data.get('splits', [])
    for split in splits:
        if split['name'] == 'train':
            original_len = len(split['shardLengths'])
            print(f"Original shard count: {original_len}")
            
            # Truncate to 100
            # We assume we have the first 100.
            # If we have less, we should count the files.
            # Let's count files to be safe.
            data_dir = os.path.dirname(info_path)
            # Files look like: fractal20220817_data-train.tfrecord-00000-of-01024
            files = [f for f in os.listdir(data_dir) if 'tfrecord' in f]
            num_shards = len(files)
            print(f"Found {num_shards} tfrecord files.")
            
            if num_shards < original_len:
                print(f"Patching dataset_info.json to use only {num_shards} shards.")
                split['shardLengths'] = split['shardLengths'][:num_shards]
                
                # We also need to update the total numBytes? 
                # TFDS might check this, but usually it just sums shardLengths for record counts.
                # numBytes is informative? Let's leave it or try to estimate.
                # TFDS might complain if file sizes don't match numBytes? 
                # Usually numBytes is total dataset size.
                # Let's try just updating shardLengths first.
            else:
                print("No patching needed (files >= metadata shards).")

    with open(info_path, 'w') as f:
        json.dump(data, f, indent=2)
    print("dataset_info.json patched successfully.")

if __name__ == "__main__":
    patch_dataset_info()
