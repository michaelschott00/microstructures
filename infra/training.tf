# NOTE: Currently broken, newest "runpod/runpod" provider version (1.0.9) gives errors, older version (1.0.8) queries non-existent endpoints

terraform {
  required_providers {
    runpod = {
      source = "runpod/runpod"
    }
  }
}

# API key can be set via environment variable RUNPOD_API_KEY
# or in the provider configuration (not shown in this example)

resource "runpod_pod" "microstructures" {
  machine_id = "your-machine-id"
  image_name = "michaelschott00/microstructures"
  gpu_count  = 1
  start_ssh  = true
}
