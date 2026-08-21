import yaml
from pathlib import Path

def load_config(config_path: str = "config.yaml") -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path.absolute()}")
    
    with open(path, "r") as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    config = load_config()
    print(f"Active Model: {config['model']['active_model']}")
    print(f"Batch Size: {config['data']['batch_size']}")