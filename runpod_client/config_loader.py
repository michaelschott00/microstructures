import os
from pathlib import Path
from typing import Any, Dict, List, TypeAlias

import yaml
from dotenv import load_dotenv

YAML_Type: TypeAlias = (
    Dict[str, "YAML_Type"] | List["YAML_Type"] | str | int | float | bool | None
)


class ConfigLoader:
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        load_dotenv()

    def load_yaml(self, filename: str) -> Dict[str, Any]:
        path = self.config_dir / filename
        with open(path) as f:
            return yaml.safe_load(f)

    def _substitute_env_vars(self, config: YAML_Type) -> YAML_Type:
        """Recursively replace ${VAR_NAME} with environment variables"""
        if isinstance(config, dict):
            return {k: self._substitute_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._substitute_env_vars(item) for item in config]
        elif isinstance(config, str):
            # Replace ${VAR_NAME} patterns
            import re

            def replace_var(match):
                var_name = match.group(1)
                value = os.getenv(var_name)
                if value is None:
                    raise ValueError(f"Environment variable '{var_name}' not found")
                return value

            return re.sub(r"\$\{([^}]+)\}", replace_var, config)
        else:
            return config

    def load_pod_config(self, pod_name: str) -> YAML_Type:
        """Load a pod-specific config"""
        pods = self.load_yaml("pods.yaml")

        if pod_name not in pods:
            raise ValueError(f"Pod '{pod_name}' not found in config")

        return self._substitute_env_vars(pods[pod_name])

    def load_network_volume_config(self, volume_name: str) -> YAML_Type:
        """Load a network-volume-specific config"""
        volumes = self.load_yaml("network_volumes.yaml")

        if volume_name not in volumes:
            raise ValueError(f"Network volume '{volume_name}' not found in config")

        return self._substitute_env_vars(volumes[volume_name])

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
