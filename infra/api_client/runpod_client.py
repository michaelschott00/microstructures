import json
import requests
from typing import Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class RunPodConfig:
    base_url: str = "https://api.runpod.io/v2"
    token: str | None = None
    timeout: int = 30
    verbose: bool = False

class RunPodClient:
    def __init__(self, config: RunPodConfig):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {config.token}",
            "Content-Type": "application/json"
        })
    
    @staticmethod
    def _pretty_json(value: str) -> str:
        try:
            return json.dumps(json.loads(value), indent=2)
        except json.JSONDecodeError:
            return value

    def _print_request(self, prepared: requests.PreparedRequest) -> None:
        print(f"Request: {prepared.method} {prepared.url}")
        print(f"Headers:\n{json.dumps(dict(prepared.headers), indent=2)}")
        body = prepared.body
        if isinstance(body, bytes):
            body = body.decode("utf-8", errors="replace")
        if isinstance(body, str) and body:
            print(f"Body:\n{self._pretty_json(body)}")

    def _print_response(self, response: requests.Response) -> None:
        print(f"Response status: {response.status_code}")
        print(f"Response headers:\n{json.dumps(dict(response.headers), indent=2)}")
        if response.text:
            print(f"Response body:\n{self._pretty_json(response.text)}")

    def _request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        url = f"{self.config.base_url}{endpoint}"
        if self.config.verbose:
            prepared = self.session.prepare_request(
                requests.Request(method, url, **kwargs)
            )
            self._print_request(prepared)
        try:
            response = self.session.request(
                method, url, timeout=self.config.timeout, **kwargs
            )
            if self.config.verbose:
                self._print_response(response)
            response.raise_for_status()
            if not response.content:
                return {}
            return response.json()
        except requests.exceptions.HTTPError as e:
            print(f"API Error: {e}")
            if e.response is not None and not self.config.verbose:
                self._print_response(e.response)
            raise
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