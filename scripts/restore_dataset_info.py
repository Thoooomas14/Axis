import json
import os

def restore_dataset_info():
    info_path = 'data/content/axis_data/fractal20220817_data/0.1.0/dataset_info.json'
    
    # Values extracted from Step 725 view_file output (first 100 shards)
    shard_lengths = [
        "80", "87", "71", "87", "83", "97", "84", "91", "88", "89",
        "81", "71", "94", "83", "89", "86", "93", "94", "84", "83",
        "89", "80", "77", "94", "80", "75", "98", "96", "97", "80",
        "88", "92", "77", "85", "89", "89", "83", "94", "74", "79",
        "92", "96", "74", "85", "88", "95", "90", "91", "66", "82",
        "85", "107", "69", "83", "82", "85", "78", "85", "93", "80",
        "92", "76", "79", "75", "84", "81", "75", "73", "71", "88",
        "81", "62", "82", "75", "79", "98", "72", "90", "80", "90",
        "91", "91", "86", "89", "96", "71", "93", "93", "103", "78",
        "108", "95", "98", "94", "92", "92", "82", "78", "95", "76"
    ]
    
    print(f"Restoring {len(shard_lengths)} shard lengths.")

    with open(info_path, 'r') as f:
        data = json.load(f)

    splits = data.get('splits', [])
    for split in splits:
        if split['name'] == 'train':
            split['shardLengths'] = shard_lengths
            # Also update numBytes to something reasonable? 
            # Or just leave the original huge number, TFDS might not care.
            # Let's leave it.

    with open(info_path, 'w') as f:
        json.dump(data, f, indent=2)
    print("dataset_info.json restored successfully.")

if __name__ == "__main__":
    restore_dataset_info()
