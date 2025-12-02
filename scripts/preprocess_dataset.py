import os
import sys
import torch
import numpy as np
import argparse
from tqdm import tqdm
import tensorflow as tf
import tensorflow_datasets as tfds
from multiprocessing import Process

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from training.utils.gemini_mock import MockGeminiPlanner

def process_shard(worker_id, num_workers, args):
    # Setup TFDS
    data_dir = args.data_dir
    builder = tfds.builder_from_directory(data_dir)
    ds = builder.as_dataset(split='train')
    
    # Shard
    if num_workers > 1:
        ds = ds.shard(num_workers, worker_id)
    
    mocker = MockGeminiPlanner()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    episode_count = 0
    
    iterator = ds
    if worker_id == 0:
        iterator = tqdm(ds, desc=f"Worker {worker_id}")
        
    for episode in iterator:
        if args.max_episodes is not None and episode_count >= args.max_episodes // num_workers:
            break
            
        steps = list(episode['steps'])
        episode_len = len(steps)
        
        if episode_len == 0:
            continue

        # 1. Instruction
        instr = "do something"
        if 'natural_language_instruction' in steps[0]['observation']:
            instr = steps[0]['observation']['natural_language_instruction'].numpy().decode('utf-8')
            
        # 3. Extract Start/End Poses for Goal
        def get_pose(step_data):
            obs = step_data['observation']
            p = np.zeros(6, dtype=np.float32)
            if 'ee_pose' in obs: p = obs['ee_pose'].numpy()
            elif 'pose' in obs: p = obs['pose'].numpy()
            # Pad to 7D
            p7 = np.zeros(7, dtype=np.float32)
            p7[:6] = p[:6] if p.shape[0] >= 6 else np.pad(p, (0, 6-p.shape[0]))
            p7[6] = 1.0
            return p7

        # 4. Sub-Task Splitting Logic (Semantic)
        grippers = []
        for s in steps:
            obs = s['observation']
            if 'gripper_closed' in obs:
                grippers.append(obs['gripper_closed'].numpy())
            else:
                grippers.append(0.0) # Default open
        
        grippers = np.array(grippers).flatten()
        
        # Detect changes using threshold
        is_closed = grippers > 0.5
        changes = is_closed[1:] != is_closed[:-1]
        change_indices = np.where(changes)[0]
        
        split_idx = -1
        
        if len(change_indices) > 0:
            idx = change_indices[0]
            split_idx = idx + 1
            if split_idx < 5 or split_idx > episode_len - 5:
                split_idx = -1
        
        goal_embs_seq = [] 
        requery_labels_seq = [] 
        
        if split_idx != -1:
            # Phase 1
            start_pose_1 = get_pose(steps[0])
            end_pose_1 = get_pose(steps[split_idx])
            goal_1 = mocker.encode_goal(start_pose=start_pose_1, end_pose=end_pose_1, instruction=instr)
            
            for t in range(split_idx + 1):
                goal_embs_seq.append(goal_1)
                requery_labels_seq.append(1.0 if t == split_idx else 0.0)
                
            # Phase 2
            start_pose_2 = get_pose(steps[split_idx + 1]) if split_idx + 1 < episode_len else end_pose_1
            end_pose_2 = get_pose(steps[-1])
            goal_2 = mocker.encode_goal(start_pose=start_pose_2, end_pose=end_pose_2, instruction=instr)
            
            for t in range(split_idx + 1, episode_len):
                goal_embs_seq.append(goal_2)
                requery_labels_seq.append(1.0 if t == episode_len - 1 else 0.0)
                
        else:
            # Single Phase
            start_pose = get_pose(steps[0])
            end_pose = get_pose(steps[-1])
            goal = mocker.encode_goal(start_pose=start_pose, end_pose=end_pose, instruction=instr)
            
            for t in range(episode_len):
                goal_embs_seq.append(goal)
                requery_labels_seq.append(1.0 if t == episode_len - 1 else 0.0)

        # 5. Process Steps
        imgs = []
        props = []
        acts = []
        
        for step_data in steps:
            # Image
            obs = step_data['observation']
            img = None
            for k in obs:
                if 'image' in k: 
                    img = obs[k]
                    break
            if img is None: img = tf.zeros((128, 128, 3), dtype=tf.uint8)
            
            # Resize/Norm
            img = tf.image.resize(img, (128, 128))
            img = tf.cast(img, tf.float32) / 255.0
            img = tf.transpose(img, [2, 0, 1]).numpy() # CHW
            imgs.append(img)
            
            # Proprio
            p = get_pose(step_data)[:6]
            g_state = 0.0
            if 'gripper_closed' in step_data['observation']:
                g_state = step_data['observation']['gripper_closed'].numpy()
            elif 'gripper_state' in step_data['observation']:
                g_state = step_data['observation']['gripper_state'].numpy()
            
            if not np.isscalar(g_state): g_state = g_state.item() if g_state.size == 1 else g_state[0]
            
            p7 = np.concatenate([p, [g_state]])
            props.append(p7)
            
            # Action
            a = np.zeros(7, dtype=np.float32)
            if 'action' in step_data:
                raw_a = step_data['action']
                if isinstance(raw_a, dict):
                    parts = []
                    if 'world_vector' in raw_a: parts.append(raw_a['world_vector'].numpy())
                    if 'rotation_delta' in raw_a: parts.append(raw_a['rotation_delta'].numpy())
                    if 'gripper_closedness_action' in raw_a:
                         g_act = raw_a['gripper_closedness_action'].numpy()
                         if np.isscalar(g_act): g_act = np.array([g_act])
                         parts.append(g_act)
                    
                    if parts: a = np.concatenate(parts, axis=-1)
                else:
                    a[:6] = raw_a.numpy()
            
            if a.shape[0] < 7: a = np.pad(a, (0, 7-a.shape[0]))
            acts.append(a[:7])
            
        data_dict = {
            'images': torch.tensor(np.array(imgs), dtype=torch.float16), 
            'proprio': torch.tensor(np.array(props), dtype=torch.float32),
            'action': torch.tensor(np.array(acts), dtype=torch.float32),
            'goal_embs': torch.tensor(np.array(goal_embs_seq), dtype=torch.float32),
            'requery_labels': torch.tensor(np.array(requery_labels_seq), dtype=torch.float32),
            'mask': torch.ones(len(imgs), dtype=torch.bool)
        }
        
        save_path = os.path.join(args.output_dir, f"episode_{worker_id}_{episode_count}.pt")
        torch.save(data_dict, save_path)
        episode_count += 1
        
    print(f"Worker {worker_id} finished. Processed {episode_count} episodes.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='fractal20220817_data')
    parser.add_argument('--data_dir', type=str, default='data/content/axis_data/fractal20220817_data/0.1.0')
    parser.add_argument('--output_dir', type=str, default='data/processed/fractal_v1')
    parser.add_argument('--max_episodes', type=int, default=None)
    parser.add_argument('--num_workers', type=int, default=4, help="Number of parallel workers")
    args = parser.parse_args()
    
    print(f"Starting preprocessing with {args.num_workers} workers...")
    
    if args.num_workers > 1:
        processes = []
        for i in range(args.num_workers):
            p = Process(target=process_shard, args=(i, args.num_workers, args))
            p.start()
            processes.append(p)
        
        for p in processes:
            p.join()
    else:
        process_shard(0, 1, args)
        
    print("All workers finished.")

if __name__ == "__main__":
    main()
