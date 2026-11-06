#!/usr/bin/env python3
"""
visual_server.py

High-performance real-time visual simulation server for the 5,000-neuron Drosophila brain.
Powered by PyTorch CUDA and Tornado WebSockets.

Streams live neural spike activity, raster data, and behavioral circuit telemetry
(locomotion steering, proboscis feeding extension) directly to an interactive WebGL/Canvas UI.

Default URL: http://localhost:8080
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

# Ensure code directory is on import path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import tornado.web
import tornado.websocket
import tornado.ioloop

from realtime_subnet_engine import RealtimeSubnetLIF, build_engine


def generate_neuron_coordinates(meta: dict) -> List[Dict]:
    """
    Generate anatomically inspired 2D coordinates [-1, 1] for all 5,000 neurons,
    clustering them by biological functional circuits (taste, feeding, locomotion, SEZ, brain lobes).
    """
    neurons = meta["neurons"]
    rng = np.random.RandomState(42)

    coords = []
    for n in neurons:
        idx = n["local_index"]
        grp = n["group"]
        lbl = n["label"]

        if grp == "p9_input":
            x = -0.65 if "left" in lbl else 0.65
            y = 0.70 + rng.uniform(-0.02, 0.02)
            color = "#00f0ff"  # Cyan
        elif grp == "sugar_grn":
            r = rng.uniform(0.0, 0.12)
            theta = rng.uniform(0, 2 * np.pi)
            x = -0.28 + r * np.cos(theta)
            y = -0.68 + r * np.sin(theta)
            color = "#00ff66"  # Emerald Green
        elif grp == "bitter_grn":
            r = rng.uniform(0.0, 0.12)
            theta = rng.uniform(0, 2 * np.pi)
            x = 0.28 + r * np.cos(theta)
            y = -0.68 + r * np.sin(theta)
            color = "#ff3333"  # Red
        elif grp == "motor_mn9":
            x = -0.10 if "left" in lbl else 0.10
            y = -0.45 + rng.uniform(-0.02, 0.02)
            color = "#ff00dd"  # Magenta
        elif grp == "descending":
            if "DNa01" in lbl:
                x = -0.30 if "left" in lbl else 0.30
                y = 0.45
            elif "DNa02" in lbl:
                x = -0.45 if "left" in lbl else 0.45
                y = 0.48
            elif "P9_oDN1" in lbl:
                x = -0.12 if "left" in lbl else 0.12
                y = 0.55
            elif "MDN" in lbl:
                x = rng.uniform(-0.15, 0.15)
                y = 0.35 + rng.uniform(-0.03, 0.03)
            else:
                x = rng.uniform(-0.2, 0.2)
                y = 0.45
            color = "#ffcc00"  # Golden Yellow
        elif grp == "command_extra":
            x = rng.uniform(-0.2, 0.2)
            y = 0.25 + rng.uniform(-0.04, 0.04)
            color = "#ff9900"  # Amber
        elif grp == "sez_interneuron":
            r_x = rng.uniform(0.0, 0.32)
            r_y = rng.uniform(0.0, 0.22)
            theta = rng.uniform(0, 2 * np.pi)
            x = r_x * np.cos(theta)
            y = -0.22 + r_y * np.sin(theta)
            color = "#7744ff"  # Indigo
        else:  # partner_interneuron
            # Distributed across bilateral brain hemispheres and central complex
            hemi = rng.choice([-1, 1, 0], p=[0.42, 0.42, 0.16])
            if hemi == 0:  # Central
                x = rng.uniform(-0.25, 0.25)
                y = rng.uniform(-0.1, 0.35)
            else:  # Left or Right lobe
                center_x = 0.52 * hemi
                center_y = 0.12
                r = rng.uniform(0.05, 0.34)
                theta = rng.uniform(0, 2 * np.pi)
                x = center_x + r * np.cos(theta)
                y = center_y + (r * 0.8) * np.sin(theta)
            color = "#446688"  # Slate Blue

        coords.append({
            "id": idx,
            "fw": n["flywire_id"],
            "grp": grp,
            "lbl": lbl,
            "x": round(float(x), 4),
            "y": round(float(y), 4),
            "c": color,
        })
    return coords


class SimulationManager:
    """Manages the real-time simulation thread, rate controls, and streaming buffers."""

    def __init__(self, engine: RealtimeSubnetLIF, meta: dict, device: str = "cuda"):
        self.engine = engine
        self.meta = meta
        self.device = engine.device
        self.num_neurons = engine.num_neurons

        self.coords = generate_neuron_coordinates(meta)

        # Simulation states
        self.is_running = True
        self.speed_factor = 1.0  # 1.0x real-time
        self.sim_time_sec = 0.0
        self.total_steps = 0

        # Dynamic Stimulation Rates (Hz)
        self.sugar_rate = 0.0
        self.bitter_rate = 0.0
        self.p9_left_rate = 0.0
        self.p9_right_rate = 0.0

        # Rate tensor
        self.rates = torch.zeros(self.num_neurons, dtype=torch.float32, device=self.device)
        self.update_stimulation_rates()

        # Telemetry metrics
        self.last_wall_time = time.perf_counter()
        self.fps = 30
        self.realtime_ratio = 1.0
        self.step_latency_us = 72.0
        self.vram_mb = 14.0

        # Frame step batch size: 30 FPS with dt=0.1ms -> ~333 steps per frame
        self.steps_per_frame = 300
        # Initialize CUDA graph for this frame block size
        if self.device.type == "cuda":
            print(f"Pre-capturing CUDA Graph for streaming block ({self.steps_per_frame} steps)...")
            self.engine.capture_cuda_graph(block_size=self.steps_per_frame)
            print("CUDA Graph ready for 60 FPS streaming!")

    def update_stimulation_rates(self):
        """Update rate tensor and engine refractory settings."""
        self.rates.zero_()
        stim_indices = []

        if self.sugar_rate > 0:
            self.rates[self.engine.sugar_indices] = self.sugar_rate
            stim_indices.extend(self.engine.sugar_indices.tolist())

        if self.bitter_rate > 0:
            self.rates[self.engine.bitter_indices] = self.bitter_rate
            stim_indices.extend(self.engine.bitter_indices.tolist())

        p9_ids = self.engine.p9_indices.tolist()
        if len(p9_ids) >= 2:
            if self.p9_left_rate > 0:
                self.rates[p9_ids[0]] = self.p9_left_rate
                stim_indices.append(p9_ids[0])
            if self.p9_right_rate > 0:
                self.rates[p9_ids[1]] = self.p9_right_rate
                stim_indices.append(p9_ids[1])

        stim_t = torch.tensor(stim_indices, dtype=torch.int64, device=self.device)
        self.engine.set_stimulated_neurons(stim_t)

    def reset(self):
        """Reset simulation state and timers."""
        self.engine.reset_state()
        self.sim_time_sec = 0.0
        self.total_steps = 0

    def step_frame(self) -> dict:
        """
        Advance simulation by one display frame (~300 steps = 30 ms biological time).
        Returns telemetry payload for WebSockets.
        """
        if not self.is_running:
            return {"paused": True, "t": round(self.sim_time_sec, 3)}

        t0 = time.perf_counter()

        # Save pre-counts
        pre_counts = self.engine.spike_counts.clone()

        # Run block
        self.engine.step_block(
            self.rates, block_steps=self.steps_per_frame, use_cuda_graph=(self.device.type == "cuda")
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize()

        t1 = time.perf_counter()
        wall_dt = max(t1 - t0, 1e-6)

        # Biological duration advanced
        frame_bio_sec = self.steps_per_frame * (self.engine.dt * 1e-3)  # e.g. 0.030 s
        self.sim_time_sec += frame_bio_sec
        self.total_steps += self.steps_per_frame

        # Compute which neurons spiked during this frame window
        frame_spikes = (self.engine.spike_counts - pre_counts).cpu().numpy()
        spiked_indices = np.where(frame_spikes > 0)[0].tolist()

        # Telemetry
        self.realtime_ratio = frame_bio_sec / wall_dt
        self.step_latency_us = (wall_dt * 1e6) / self.steps_per_frame
        if self.device.type == "cuda":
            self.vram_mb = torch.cuda.memory_allocated() / 1024 / 1024

        # Key behavioral firing rates (Hz over the frame)
        rates_hz = {}
        for lbl, idx in self.engine.descending_map.items():
            rates_hz[lbl] = round(float(frame_spikes[idx] / frame_bio_sec), 1)

        for lbl, idx in self.engine.mn9_map.items():
            rates_hz[lbl] = round(float(frame_spikes[idx] / frame_bio_sec), 1)

        # Average group rates
        sugar_mean = float(np.mean(frame_spikes[self.engine.sugar_indices.cpu().numpy()]) / frame_bio_sec)
        bitter_mean = float(np.mean(frame_spikes[self.engine.bitter_indices.cpu().numpy()]) / frame_bio_sec)

        return {
            "t": round(self.sim_time_sec, 3),
            "steps": self.total_steps,
            "spikes": spiked_indices,
            "spikes_count": len(spiked_indices),
            "rt_ratio": round(self.realtime_ratio, 2),
            "latency_us": round(self.step_latency_us, 1),
            "vram_mb": round(self.vram_mb, 1),
            "rates": rates_hz,
            "sugar_rate": round(sugar_mean, 1),
            "bitter_rate": round(bitter_mean, 1),
        }


# Connected WebSocket clients
ws_clients: Set[tornado.websocket.WebSocketHandler] = set()
sim_manager: Optional[SimulationManager] = None


class IndexHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("index.html")


class WebSocketHandler(tornado.websocket.WebSocketHandler):
    def check_origin(self, origin):
        return True

    def open(self):
        ws_clients.add(self)
        # Send initial setup package
        init_payload = {
            "type": "init",
            "num_neurons": sim_manager.num_neurons,
            "coords": sim_manager.coords,
            "groups": sim_manager.engine.groups,
        }
        self.write_message(json.dumps(init_payload))

    def on_close(self):
        ws_clients.discard(self)

    def on_message(self, message):
        try:
            data = json.loads(message)
            action = data.get("action")

            if action == "set_rates":
                sim_manager.sugar_rate = float(data.get("sugar", sim_manager.sugar_rate))
                sim_manager.bitter_rate = float(data.get("bitter", sim_manager.bitter_rate))
                sim_manager.p9_left_rate = float(data.get("p9_left", sim_manager.p9_left_rate))
                sim_manager.p9_right_rate = float(data.get("p9_right", sim_manager.p9_right_rate))
                sim_manager.update_stimulation_rates()

            elif action == "pause":
                sim_manager.is_running = False

            elif action == "resume":
                sim_manager.is_running = True

            elif action == "reset":
                sim_manager.reset()

        except Exception as e:
            print(f"WebSocket message error: {e}")


def broadcast_simulation_step():
    """Periodic callback to advance the simulation and push frames to WebSockets."""
    if not ws_clients or sim_manager is None:
        return

    frame_data = sim_manager.step_frame()
    msg = json.dumps({"type": "frame", "data": frame_data})

    for client in list(ws_clients):
        try:
            client.write_message(msg)
        except Exception:
            ws_clients.discard(client)


def create_app(static_dir: str):
    return tornado.web.Application(
        [
            (r"/", IndexHandler),
            (r"/ws", WebSocketHandler),
            (r"/(.*)", tornado.web.StaticFileHandler, {"path": static_dir}),
        ],
        template_path=static_dir,
        static_path=static_dir,
        debug=False,
    )


def main():
    parser = argparse.ArgumentParser(description="Real-Time Drosophila Visual Simulation Server")
    parser.add_argument("--port", type=int, default=8080, help="Server port (default: 8080)")
    parser.add_argument("--device", type=str, default="cuda", help="Execution device (cuda/cpu)")
    parser.add_argument("--subnetwork-dir", type=str, default="data/subnetwork_5k", help="Subnetwork directory")
    args = parser.parse_args()

    global sim_manager

    print("=" * 75)
    print("Initializing Drosophila Connectome Real-Time Visual Server")
    print(f"Device: {args.device} | Port: {args.port}")
    print("=" * 75)

    engine, meta = build_engine(subnetwork_dir=args.subnetwork_dir, device=args.device)
    sim_manager = SimulationManager(engine=engine, meta=meta, device=args.device)

    current_dir = Path(__file__).resolve().parent
    static_dir = current_dir / "static"

    app = create_app(str(static_dir))
    app.listen(args.port, address="0.0.0.0")

    # Start 30 FPS periodic simulation broadcast loop (every 33 ms)
    tornado.ioloop.PeriodicCallback(broadcast_simulation_step, 33).start()

    print(f"\n Visual Simulation is LIVE!")
    print(f"  -> Open in your browser: http://localhost:{args.port}")
    print(f"  -> WSL / LAN Address    : http://0.0.0.0:{args.port}")
    print("Press Ctrl+C to terminate.")
    print("=" * 75)

    try:
        tornado.ioloop.IOLoop.current().start()
    except KeyboardInterrupt:
        print("\nStopping visual simulation server...")


if __name__ == "__main__":
    main()
