import torch
import os
import json
import matplotlib.pyplot as plt
from datetime import datetime

class ExperimentTracker:
    """
    Utility to handle checkpointing, logging, and metrics for academic reporting.
    Supports pause/resume and metadata preservation.
    """
    def __init__(self, experiment_name, base_dir="outputs/models"):
        self.experiment_name = experiment_name
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_dir = os.path.join(base_dir, f"{experiment_name}_{self.run_id}")
        os.makedirs(self.exp_dir, exist_ok=True)
        
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "metrics": {},
            "metadata": {
                "experiment_name": experiment_name,
                "run_id": self.run_id,
                "start_time": str(datetime.now())
            }
        }
        
    def save_checkpoint(self, model, optimizer, epoch, val_loss, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'history': self.history
        }
        
        # Save regular epoch checkpoint
        path = os.path.join(self.exp_dir, "latest_checkpoint.pt")
        torch.save(checkpoint, path)
        
        # Save best model separately
        if is_best:
            best_path = os.path.join(self.exp_dir, "best_model.pt")
            torch.save(checkpoint, best_path)
            
        # Save metrics as JSON for easy plotting later
        with open(os.path.join(self.exp_dir, "history.json"), "w") as f:
            json.dump(self.history, f, indent=4)

    def log_epoch(self, epoch, train_loss, val_loss, **kwargs):
        self.history["train_loss"].append(train_loss)
        self.history["val_loss"].append(val_loss)
        for k, v in kwargs.items():
            if k not in self.history["metrics"]:
                self.history["metrics"][k] = []
            self.history["metrics"][k].append(v)
            
    @staticmethod
    def load_checkpoint(checkpoint_path, model, optimizer=None):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch'], checkpoint['val_loss'], checkpoint['history']

    def plot_curves(self):
        """Generates loss/metric curves for the master's report."""
        plt.figure(figsize=(10, 5))
        plt.plot(self.history["train_loss"], label="Train Loss")
        plt.plot(self.history["val_loss"], label="Val Loss")
        plt.title(f"Learning Curves: {self.experiment_name}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(self.exp_dir, "learning_curves.png"))
        plt.close()
