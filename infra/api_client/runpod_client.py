import requests
from typing import Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class RunPodConfig:
    base_url: str = "https://api.runpod.io/v2"
    token: str | None = None
    timeout: int = 30

class RunPodClient:
    def __init__(self, config: RunPodConfig):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {config.token}",
            "Content-Type": "application/json"
        })
    
    def _request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        url = f"{self.config.base_url}{endpoint}"
        try:
            response = self.session.request(
                method, url, timeout=self.config.timeout, **kwargs
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"API Error: {e}")
            raise
    
    def create_pod(self, pod_config: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("POST", "/pods", json=pod_config)
    
    def get_pod(self, pod_id: str) -> Dict[str, Any]:
        return self._request("GET", f"/pods/{pod_id}")
    
    def list_pods(self) -> Dict[str, Any]:
        return self._request("GET", "/pods")
    
    def delete_pod(self, pod_id: str) -> Dict[str, Any]:
        return self._request("DELETE", f"/pods/{pod_id}")

    def create_network_volume(self, volume_config: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("POST", "/network-volumes", json=volume_config)

    def get_network_volume(self, volume_id: str) -> Dict[str, Any]:
        return self._request("GET", f"/network-volumes/{volume_id}")

    def list_network_volumes(self) -> Dict[str, Any]:
        return self._request("GET", "/network-volumes")

    def delete_network_volume(self, volume_id: str) -> Dict[str, Any]:
        return self._request("DELETE", f"/network-volumes/{volume_id}")

    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.session.close()