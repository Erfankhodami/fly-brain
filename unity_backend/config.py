"""
unity_backend/config.py

Configuration parameters for the Drosophila connectome Unity backend.
"""

from pathlib import Path

# Repository root directory
REPO_ROOT = Path(__file__).resolve().parent.parent

# Subnetwork data paths
SUBNETWORK_DIR = REPO_ROOT / "data" / "subnetwork_5k"
CSR_PATH = SUBNETWORK_DIR / "subnetwork_csr.pt"
META_PATH = SUBNETWORK_DIR / "subnetwork_meta.json"

# Networking settings
HOST = "0.0.0.0"
PORT = 8080
WS_ROUTE = "/game_ws"
HTTP_HEALTH_ROUTE = "/health"

# Physics & Simulation Frame Timing
PHYSICS_FPS = 60          # 60 Hz physics update rate (16.6 ms per tick)
DT_NEURON_MS = 0.1        # 0.1 ms discrete simulation timestep
STEPS_PER_TICK = 166      # 16.6 ms / 0.1 ms ~ 166 simulation steps per physics frame

# Motor Decoding Scaling Constants
MAX_FORWARD_THRUST_HZ = 16.0    # P9_oDN1 smoothed rate corresponding to 100% forward speed
MAX_YAW_TORQUE_HZ = 10.0        # DNa02 differential (R - L) corresponding to 100% turning torque
MN9_EATING_THRESHOLD_HZ = 20.0  # MN9 proboscis motor rate required to enter eating state
MN9_MAX_EXTENSION_HZ = 65.0     # MN9 rate corresponding to 100% proboscis extension

# Sensory Encoding Constants
BASE_EXPLORATORY_RATE = 35.0    # Spontaneous search rate (Hz) when no target is visible
BASE_VISUAL_DRIVE_RATE = 50.0   # Minimum visual drive when target is spotted
MAX_VISUAL_DRIVE_RATE = 120.0   # Maximum rate driven by visual target intensity
SUGAR_CONTACT_RATE = 250.0      # Stimulation rate (Hz) applied to Sugar GRNs on contact
BITTER_CONTACT_RATE = 200.0     # Stimulation rate (Hz) applied to Bitter GRNs on contact
