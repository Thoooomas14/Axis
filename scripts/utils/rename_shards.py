import os

def rename_shards():
    data_dir = 'data/content/axis_data/fractal20220817_data/0.1.0'
    
    if not os.path.exists(data_dir):
        print(f"Error: {data_dir} not found.")
        return

    files = [f for f in os.listdir(data_dir) if 'tfrecord' in f]
    print(f"Found {len(files)} files.")

    count = 0
    for filename in files:
        # Expected format: fractal20220817_data-train.tfrecord-00000-of-01024
        if 'of-01024' in filename:
            new_filename = filename.replace('of-01024', 'of-00100')
            old_path = os.path.join(data_dir, filename)
            new_path = os.path.join(data_dir, new_filename)
            
            os.rename(old_path, new_path)
            count += 1
            
    print(f"Renamed {count} files.")

if __name__ == "__main__":
    rename_shards()
