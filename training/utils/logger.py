import os
import csv
import matplotlib.pyplot as plt
import pandas as pd

class TrainingLogger:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, 'training_log.csv')
        self.plot_path = os.path.join(log_dir, 'training_plot.png')
        
        # Initialize CSV if not exists
        if not os.path.exists(self.log_path):
            with open(self.log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['step', 'epoch', 'loss', 'action_loss', 'requery_loss'])

    def log_step(self, step, epoch, loss, action_loss, requery_loss):
        with open(self.log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([step, epoch, loss, action_loss, requery_loss])

    def plot_progress(self):
        try:
            df = pd.read_csv(self.log_path)
            if len(df) < 2: return
            
            plt.figure(figsize=(12, 6))
            
            # Plot Total Loss
            plt.subplot(1, 2, 1)
            plt.plot(df['step'], df['loss'], label='Total Loss', color='blue')
            plt.xlabel('Step')
            plt.ylabel('Loss')
            plt.title('Training Loss')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Plot Components
            plt.subplot(1, 2, 2)
            plt.plot(df['step'], df['action_loss'], label='Action Loss', color='green', alpha=0.7)
            plt.plot(df['step'], df['requery_loss'], label='Requery Loss', color='red', alpha=0.7)
            plt.xlabel('Step')
            plt.title('Loss Components')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(self.plot_path)
            plt.close()
            
        except Exception as e:
            print(f"Error plotting progress: {e}")
