"""
unity_backend

Dedicated Python backend interfacing the 5,000-neuron Drosophila connectome subnetwork
with Unity 3D game engines via high-speed WebSockets.
"""

from . import config
from .sensory_encoder import SensoryEncoder
from .motor_decoder import MotorDecoder
from .game_server import UnityGameServerManager, create_game_app

__all__ = [
    "config",
    "SensoryEncoder",
    "MotorDecoder",
    "UnityGameServerManager",
    "create_game_app",
]
