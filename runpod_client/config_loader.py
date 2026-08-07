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
        """Load a pod-specific config"""
        pods = self.load_yaml("pods.yaml")

        if pod_name not in pods:
            raise ValueError(f"Pod '{pod_name}' not found in config")

        return pods[pod_name]
    
    def load_network_volume_config(self, volume_name: str) -> Dict[str, Any]:
        """Load a network-volume-specific config"""
        volumes = self.load_yaml("network_volumes.yaml")

        if volume_name not in volumes:
            raise ValueError(f"Network volume '{volume_name}' not found in config")

        return volumes[volume_name]

    def get_api_token(self) -> str:
        """Load from env or secrets file"""
        token = os.getenv("RUNPOD_API_KEY")
        if not token:
            try:
                with open(self.config_dir / "secrets.env") as f:
                    for line in f:
                        if line.startswith("RUNPOD_API_KEY="):
                            token = line.split("=", 1)[1].strip()
                            break
            except FileNotFoundError:
                pass
        
        if not token:
            raise ValueError("RUNPOD_API_KEY not found")
        return token