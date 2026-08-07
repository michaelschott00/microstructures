from typing import Any, Dict, List

from runpod_client.config_loader import ConfigLoader
from runpod_client.runpod_client import RunPodClient, RunPodConfig


class NetworkVolumeManager:
    def __init__(self, config_dir: str = "config", verbose: bool = False):
        self.loader = ConfigLoader(config_dir)
        token = self.loader.get_api_token()
        self.client = RunPodClient(RunPodConfig(token=token, verbose=verbose))

    def create_from_config(self, volume_name: str) -> str:
        """Create network volume from config, return volume ID"""
        volume_config = self.loader.load_network_volume_config(volume_name)
        result = self.client.create_network_volume(volume_config)

        if "id" in result:
            print(f"✓ Created network volume with ID: {result['id']}")
            return result["id"]
        else:
            print(f"✗ Failed to create network volume: {result}")
            raise RuntimeError(result.get("error", "Unknown error"))

    def status(self, volume_id: str) -> Dict[str, Any]:
        return self.client.get_network_volume(volume_id)

    def cleanup(self, volume_ids: List[str]):
        """Delete multiple network volumes"""
        for volume_id in volume_ids:
            try:
                self.client.delete_network_volume(volume_id)
                print(f"✓ Deleted network volume {volume_id}")
            except Exception as e:
                print(f"✗ Failed to delete {volume_id}: {e}")
