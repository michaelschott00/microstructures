import yaml
import os
from typing import Dict, Any
from pathlib import Path

class ConfigLoader:
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
    
    def load_yaml(self, filename: str) -> Dict[str, Any]:
        path = self.config_dir / filename
        with open(path) as f:
            return yaml.safe_load(f)
    
    def load_pod_config(self, pod_name: str) -> Dict[str, Any]:
        """Merge defaults with pod-specific config"""
        defaults = self.load_yaml("defaults.yaml")
        pods = self.load_yaml("pods.yaml")
        
        if pod_name not in pods:
            raise ValueError(f"Pod '{pod_name}' not found in config")
        
        # Deep merge
        config = {**defaults, **pods[pod_name]}
        return config
    
    def load_network_volume_config(self, filename: str = "network_volume.yaml") -> Dict[str, Any]:
        """Load a network volume config file"""
        return self.load_yaml(filename)

    def get_api_token(self) -> str:
        """Load from env or secrets file"""
        token = os.getenv("RUNPOD_API_TOKEN")
        if not token:
            try:
                with open(self.config_dir / "secrets.env") as f:
                    for line in f:
                        if line.startswith("RUNPOD_API_TOKEN="):
                            token = line.split("=", 1)[1].strip()
                            break
            except FileNotFoundError:
                pass
        
        if not token:
            raise ValueError("RUNPOD_API_TOKEN not found")
        return token