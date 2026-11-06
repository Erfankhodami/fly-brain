#!/usr/bin/env python3
"""
unity_backend/game_server.py

Dedicated high-speed WebSocket backend connecting Unity 3D game characters
to the real-time 5,000-neuron Drosophila brain running on NVIDIA RTX GPUs.

Endpoints:
- WebSocket: ws://0.0.0.0:8080/game_ws
- HTTP:      http://0.0.0.0:8080/health
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional, Set

# Add repository root and code/ to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "code"))

import numpy as np
import torch
import tornado.web
import tornado.websocket
import tornado.ioloop

from realtime_subnet_engine import RealtimeSubnetLIF, build_engine
from unity_backend import config
from unity_backend.sensory_encoder import SensoryEncoder
from unity_backend.motor_decoder import MotorDecoder


class UnityGameServerManager:
    """Manages the GPU connectome engine and Unity closed-loop execution."""

    def __init__(self, device: str = "cuda"):
        print(f"Loading 5,000-neuron subnetwork on device: {device}...")
        self.engine, self.meta = build_engine(
            subnetwork_dir=str(config.SUBNETWORK_DIR), device=device
        )
        self.device = self.engine.device
        self.encoder = SensoryEncoder(self.engine)
        self.decoder = MotorDecoder(self.engine)

        self.steps_per_tick = config.STEPS_PER_TICK  # 166 steps = 16.6 ms (60 FPS)
        self.tick_bio_sec = self.steps_per_tick * (self.engine.dt * 1e-3)

        # Pre-capture CUDA Graph for 60 FPS tick execution
        if self.device.type == "cuda":
            print(f"Pre-capturing CUDA Graph for Unity physics tick ({self.steps_per_tick} steps)...")
            self.engine.capture_cuda_graph(block_size=self.steps_per_tick)
            print("CUDA Graph successfully captured!")

        self.last_sensory_payload = {}
        self.total_frames = 0
        self.total_sim_time = 0.0

    def step_physics_tick(self, sensory_data: dict) -> dict:
        """
        Ingest Unity sensory observation, step the connectome for 16.6 ms,
        and return the decoded motor forces.
        """
        t0 = time.perf_counter()

        # 1. Sensory Ingestion -> Poisson Firing Rates
        rates, stim_indices = self.encoder.encode(sensory_data)
        stim_tensor = torch.tensor(stim_indices, dtype=torch.int64, device=self.device)
        self.engine.set_stimulated_neurons(stim_tensor)

        # 2. Connectome Simulation Step
        pre_counts = self.engine.spike_counts.clone()
        self.engine.step_block(
            rates,
            block_steps=self.steps_per_tick,
            use_cuda_graph=(self.device.type == "cuda"),
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize()

        # 3. Extract Observed Spikes
        frame_spikes = (self.engine.spike_counts - pre_counts).cpu().numpy()

        # 4. Motor Output Decoding
        motor_payload = self.decoder.decode(frame_spikes, duration_sec=self.tick_bio_sec)

        t1 = time.perf_counter()
        compute_ms = (t1 - t0) * 1000.0

        self.total_frames += 1
        self.total_sim_time += self.tick_bio_sec

        motor_payload["timestamp"] = round(self.total_sim_time, 3)
        motor_payload["frame_idx"] = self.total_frames
        motor_payload["compute_ms"] = round(compute_ms, 2)

        return motor_payload


# Global manager and client registry
manager: Optional[UnityGameServerManager] = None
connected_game_clients: Set[tornado.websocket.WebSocketHandler] = set()


class HealthHandler(tornado.web.RequestHandler):
    def get(self):
        self.set_header("Content-Type", "application/json")
        self.write(
            json.dumps(
                {
                    "status": "online",
                    "neurons": manager.engine.num_neurons if manager else 5000,
                    "device": str(manager.device) if manager else "cuda",
                    "tick_hz": config.PHYSICS_FPS,
                    "connected_clients": len(connected_game_clients),
                }
            )
        )


class GameWebSocketHandler(tornado.websocket.WebSocketHandler):
    def check_origin(self, origin):
        return True

    def open(self):
        connected_game_clients.add(self)
        print(f"[Unity WebSocket] New game client connected from {self.request.remote_ip}")
        # Send initial handshake
        self.write_message(
            json.dumps(
                {
                    "type": "handshake",
                    "message": "Connected to Drosophila 5,000-Neuron GPU Backend",
                    "fps": config.PHYSICS_FPS,
                    "dt_ms": config.DT_NEURON_MS,
                }
            )
        )

    def on_close(self):
        connected_game_clients.discard(self)
        print(f"[Unity WebSocket] Game client disconnected.")

    def on_message(self, message):
        """
        Handle incoming sensory telemetry from Unity, step the connectome,
        and return the motor forces immediately.
        """
        try:
            sensory_data = json.loads(message)
            motor_response = manager.step_physics_tick(sensory_data)
            self.write_message(json.dumps(motor_response))
        except Exception as e:
            print(f"[Unity WebSocket Error] {e}")


def create_game_app():
    return tornado.web.Application(
        [
            (config.HTTP_HEALTH_ROUTE, HealthHandler),
            (config.WS_ROUTE, GameWebSocketHandler),
        ],
        debug=False,
    )


def main():
    parser = argparse.ArgumentParser(description="Unity Connectome Backend Server")
    parser.add_argument("--port", type=int, default=config.PORT, help=f"Server port (default: {config.PORT})")
    parser.add_argument("--device", type=str, default="cuda", help="Execution device (cuda/cpu)")
    args = parser.parse_args()

    global manager

    print("=" * 75)
    print("Drosophila Connectome - Dedicated Unity 3D Game Backend")
    print(f"Device: {args.device} | Port: {args.port} | Route: {config.WS_ROUTE}")
    print("=" * 75)

    manager = UnityGameServerManager(device=args.device)

    app = create_game_app()
    app.listen(args.port, address=config.HOST)

    print("\n[READY] Game Backend Server is listening for Unity connections:")
    print(f"  -> WebSocket URL : ws://localhost:{args.port}{config.WS_ROUTE}")
    print(f"  -> Health Check  : http://localhost:{args.port}{config.HTTP_HEALTH_ROUTE}")
    print("Press Ctrl+C to terminate.")
    print("=" * 75)

    try:
        tornado.ioloop.IOLoop.current().start()
    except KeyboardInterrupt:
        print("\nShutting down game server...")


if __name__ == "__main__":
    main()
