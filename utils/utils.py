import pickle
import json

from datasets.datasets import CustomDatasetWrapper

def load_targets(data: CustomDatasetWrapper, from_dir: str = None):
    if from_dir is None:
        from_dir = f"logdir/cached_targets_{data.classification_mode}.pkl"
    
    try:
        with open(from_dir, "rb") as f:
            targets = pickle.load(f)
    except FileNotFoundError:
        targets = [t for _, t in data]
        # Cache the targets for next time
        with open(from_dir, "wb") as f:
            pickle.dump(targets, f)


def load_hyperparameter(classification_mode, filename: str = None):
    
    if filename is None:
        filename = f"hyperparameters_{classification_mode}.json"
    
    with open(filename, 'r') as f:
        test_config = json.load(f)
    return test_config
