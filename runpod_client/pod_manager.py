from typing import Any, Dict, List

from runpod_client.config_loader import ConfigLoader
from runpod_client.runpod_client import RunPodClient, RunPodConfig


class PodManager:
    def __init__(self, config_dir: str = "config", verbose: bool = False):
        self.loader = ConfigLoader(config_dir)
        token = self.loader.get_api_token()
        self.client = RunPodClient(RunPodConfig(token=token, verbose=verbose))

    def create_from_config(self, pod_name: str) -> str:
        """Create pod from config file, return pod ID"""
        pod_config = self.loader.load_pod_config(pod_name)
        result = self.client.create_pod(pod_config)

        if "id" in result:
            print(f"✓ Created pod '{pod_name}' with ID: {result['id']}")
            return result["id"]
        else:
            print(f"✗ Failed to create pod: {result}")
            raise RuntimeError(result.get("error", "Unknown error"))

    def status(self, pod_id: str) -> Dict[str, Any]:
        return self.client.get_pod(pod_id)

    def cleanup(self, pod_ids: List[str]):
        """Delete multiple pods"""
        for pod_id in pod_ids:
            try:
                self.client.delete_pod(pod_id)
                print(f"✓ Deleted pod {pod_id}")
            except Exception as e:
                print(f"✗ Failed to delete {pod_id}: {e}")
