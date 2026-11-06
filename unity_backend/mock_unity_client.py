#!/usr/bin/env python3
"""
unity_backend/mock_unity_client.py

Stand-alone 3D virtual environment simulator that mimics a Unity game client.
Tests the entire closed-loop sensorimotor cycle with the Drosophila connectome server.

Scenario:
- Fly starts at position (0.0, 0.0) facing North (+Z).
- A sugar cube is placed at target position (4.0, 4.0).
- The fly uses compound eye vision to spot the sugar, steers towards it,
  and extends its proboscis to eat upon contact!
"""

import argparse
import asyncio
import json
import math
import sys
import time

try:
    import tornado.websocket
    from tornado.ioloop import IOLoop
except ImportError:
    print("Error: Tornado required for mock client.")
    sys.exit(1)


class VirtualFly3D:
    """Simulates 3D kinematic physics and compound eye vision sensors."""

    def __init__(self, target_x: float = 4.0, target_z: float = 4.0):
        # Fly 2D/3D state
        self.x = 0.0
        self.z = 0.0
        self.heading_deg = 0.0  # 0 deg = +Z (North), 90 deg = +X (East)
        self.speed = 0.0

        # Sugar Target Position
        self.sugar_x = target_x
        self.sugar_z = target_z

        # Contact radius
        self.contact_dist = 0.35

    def get_sensory_reading(self) -> dict:
        """Calculate distance, relative visual azimuth, and left/right eye intensities."""
        dx = self.sugar_x - self.x
        dz = self.sugar_z - self.z
        distance = math.hypot(dx, dz)

        # Angle from fly to sugar in world coordinates
        angle_to_sugar = math.degrees(math.atan2(dx, dz))

        # Relative angle to fly's current heading (-180 to +180 deg)
        rel_angle = (angle_to_sugar - self.heading_deg + 180) % 360 - 180

        # Bilateral compound eye vision model:
        # Left eye sensitive to [-120 deg to +10 deg] (peak at -35 deg)
        # Right eye sensitive to [-10 deg to +120 deg] (peak at +35 deg)
        left_receptive = max(0.0, 1.0 - abs(rel_angle - (-35.0)) / 90.0)
        right_receptive = max(0.0, 1.0 - abs(rel_angle - (35.0)) / 90.0)

        # Distance factor (1.0 up close, gentle falloff with distance)
        dist_factor = max(0.4, 1.0 - (distance / 12.0))

        left_intensity = left_receptive * dist_factor
        right_intensity = right_receptive * dist_factor

        is_touching = distance <= self.contact_dist

        return {
            "vision": {
                "target_visible": True,
                "left_eye_intensity": round(left_intensity, 3),
                "right_eye_intensity": round(right_intensity, 3),
                "target_azimuth_deg": round(rel_angle, 1),
                "target_distance": round(distance, 2),
            },
            "contact_taste": {
                "touching_sugar": is_touching,
                "touching_bitter": False,
            },
        }

    def update_physics(self, thrust: float, yaw_torque: float, is_eating: bool, dt: float = 0.016):
        """Update virtual position and rotation based on brain motor outputs."""
        if is_eating:
            self.speed = 0.0
            return

        # Steering: Apply yaw torque
        max_turn_deg_per_sec = 180.0
        self.heading_deg += yaw_torque * max_turn_deg_per_sec * dt
        self.heading_deg = (self.heading_deg + 180) % 360 - 180

        # Locomotion: Forward speed
        max_speed = 2.5  # units per sec
        self.speed = thrust * max_speed

        rad = math.radians(self.heading_deg)
        self.x += self.speed * math.sin(rad) * dt
        self.z += self.speed * math.cos(rad) * dt


async def run_client(server_url: str):
    print("=" * 70)
    print("Drosophila Connectome - Virtual 3D Unity Client Mock")
    print(f"Connecting to: {server_url}")
    print("=" * 70)

    try:
        ws = await tornado.websocket.websocket_connect(server_url)
        print("[CONNECTED] Connected to GPU Connectome Game Server!\n")
    except Exception as e:
        print(f"[ERROR] Failed to connect to server: {e}")
        print("Ensure unity_backend/game_server.py is running first.")
        return

    # Wait for handshake
    msg = await ws.read_message()
    handshake = json.loads(msg)
    print(f"Handshake: {handshake.get('message')} (Tick: {handshake.get('fps')} Hz)\n")

    fly = VirtualFly3D(target_x=3.5, target_z=3.5)
    print(f"Goal: Navigate from (0.0, 0.0) to Sugar at ({fly.sugar_x}, {fly.sugar_z})\n")

    step = 0
    t_start = time.perf_counter()

    try:
        while True:
            # 1. Gather Sensory Data from 3D Environment
            sensory = fly.get_sensory_reading()
            await ws.write_message(json.dumps(sensory))

            # 2. Receive Motor Outputs from the Connectome Brain
            resp = await ws.read_message()
            if resp is None:
                break
            motor = json.loads(resp)

            locomotion = motor.get("locomotion", {})
            feeding = motor.get("feeding", {})
            thrust = locomotion.get("forward_thrust", 0.0)
            torque = locomotion.get("yaw_torque", 0.0)
            proboscis = feeding.get("proboscis_extension", 0.0)
            is_eating = feeding.get("is_eating", False)

            # 3. Update Fly Physics in 3D Space
            fly.update_physics(thrust, torque, is_eating, dt=0.016)

            step += 1
            dist = sensory["vision"]["target_distance"]
            azimuth = sensory["vision"]["target_azimuth_deg"]

            if step % 10 == 0 or is_eating:
                state_str = "🍽️ [FEEDING / PROBOSCIS EXTENDED]" if is_eating else "🚶 [WALKING & STEERING]"
                print(
                    f"Step {step:4d} | Pos: ({fly.x:5.2f}, {fly.z:5.2f}) | "
                    f"Dist: {dist:4.2f}m | Azimuth: {azimuth:5.1f}° | "
                    f"Thrust: {thrust:4.2f} | Turn: {torque:+5.2f} | {state_str}"
                )

            if is_eating:
                print("\n" + "=" * 70)
                print("🎯 SUCCESS! The virtual fly reached the sugar cube and is eating!")
                print(f"Final Position: ({fly.x:.2f}, {fly.z:.2f})")
                print(f"MN9 Proboscis Extension: {proboscis * 100:.1f}%")
                print(f"Total time: {time.perf_counter() - t_start:.2f} s")
                print("=" * 70)
                break

            await asyncio.sleep(0.016)

    except KeyboardInterrupt:
        print("\nStopping mock client...")
    finally:
        ws.close()


def main():
    parser = argparse.ArgumentParser(description="Mock Unity 3D Fly Client")
    parser.add_argument(
        "--url",
        type=str,
        default="ws://127.0.0.1:8080/game_ws",
        help="Backend WebSocket URL (default: ws://127.0.0.1:8080/game_ws)",
    )
    args = parser.parse_args()

    IOLoop.current().run_sync(lambda: run_client(args.url))


if __name__ == "__main__":
    main()
