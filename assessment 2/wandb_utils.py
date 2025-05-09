import wandb
import os

def init_wandb(config, mode):
    """Initializes a new W&B run."""
    run_name = f"mode_{mode}_u{config.num_unit}_lr{config.learning_rate}_ft_{config.feature_type}"
        
    wandb.init(
        project="hog-training-job",  # W&B project name
        name=run_name,               # Name for the run
        config=vars(config),         # Log all configuration parameters
        sync_tensorboard=True,       # Automatically sync TensorBoard logs (from ./logs)
        reinit=True,                 # Allow reinitialization for multiple calls in one script if needed (e.g. train then test)

    )
    print(f"W&B run initialized: {run_name}")

def log_test_metrics(test_loss, test_accuracy):
    """Logs test metrics to W&B."""
    wandb.log({
        "final_test_loss": test_loss,
        "final_test_accuracy": test_accuracy
    })
    print(f"Logged test metrics to W&B: Loss={test_loss}, Accuracy={test_accuracy}")

def save_model_artifact(file_path):
    """Saves a file (e.g., a model) as a W&B artifact."""
    if os.path.exists(file_path):
        wandb.save(file_path, base_path=os.path.dirname(file_path))
        print(f"Saved model artifact to W&B: {file_path}")
    else:
        print(f"Warning: Model file not found at {file_path}, not saving to W&B.")

def finish_wandb():
    """Finishes the W&B run."""
    wandb.finish()
    print("W&B run finished.") 