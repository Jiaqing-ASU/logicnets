import os
import yaml
import torch
import random
import numpy as np

from dataset import HGCalAutoencoderDataset

ENSEMBLING_METHODS = ["voting", "snapshot", "fge", "adaboost", "bagging"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_data(hparams_config: str, data_path: str, data_dir: str, process_data=False, num_workers=8, save_dir="./autoencoder_test", experiment_name="autoencoder"):
    with open(hparams_config, "r") as f:
        config = yaml.safe_load(f)
    # Set random seeds
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
    print(f"Global seed set to {seed}!")

    # Create experiment directory
    experiment_dir = os.path.join(save_dir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Fetch datasets
    dataset = {}
    dataset["train"] = HGCalAutoencoderDataset(
        data_path, 
        data_dir=data_dir, 
        process_data=process_data, # Only need to pass once
        split="train",
    ) 
    dataset["valid"] = HGCalAutoencoderDataset(
        data_path, 
        data_dir=data_dir, 
        split="test",
    ) 
    dataset["test"] = HGCalAutoencoderDataset(
        data_path, 
        data_dir=data_dir, 
        split="test",
    )
    
    train_loader = torch.utils.data.DataLoader(
        dataset["train"], 
        num_workers=num_workers,
        batch_size=config["batch_size"], 
        pin_memory=True,
        shuffle=False, 
    )
    
    test_loader = torch.utils.data.DataLoader(
        dataset["test"], 
        num_workers=num_workers, 
        batch_size=config["batch_size"], 
        pin_memory=True,
        shuffle=False, 
    )

    print("Data loaded successfully.")
    return train_loader, test_loader
