"""
unity_backend/motor_decoder.py

Decodes spiking neural activity of descending command neurons and feeding motor neurons
into 3D physics variables (thrust, torque, proboscis extension, animation states) for Unity.
"""

from typing import Dict
import numpy as np

from . import config


class MotorDecoder:
    """Decodes connectome spike counts into physical game motion commands."""

    def __init__(self, engine):
        self.engine = engine
        self.descending_map = engine.descending_map
        self.mn9_map = engine.mn9_map

        # Physiological neuromuscular low-pass filter (tau = 80 ms)
        # Smooths discrete single-spike pulses into continuous muscular forces
        self.tau_muscle = 0.080  # 80 ms muscle activation constant
        self.smooth_odn = 0.0
        self.smooth_dna_l = 0.0
        self.smooth_dna_r = 0.0
        self.smooth_mn9 = 0.0
        self.smooth_mdn = 0.0

    def reset(self):
        """Reset muscle activation dynamics."""
        self.smooth_odn = 0.0
        self.smooth_dna_l = 0.0
        self.smooth_dna_r = 0.0
        self.smooth_mn9 = 0.0
        self.smooth_mdn = 0.0

    def decode(self, frame_spikes: np.ndarray, duration_sec: float) -> dict:
        """
        Convert spike counts observed during the frame window into game physics forces.

        Args:
            frame_spikes: 1D numpy array of spike counts for each of the 5,000 neurons.
            duration_sec: Real-world / biological duration of the simulated frame.

        Returns:
            Dictionary matching the Unity MotorData JSON schema.
        """
        dt = max(duration_sec, 1e-6)
        alpha = dt / (self.tau_muscle + dt)

        # 1. Forward Walking Command: P9_oDN1 (left + right)
        odn_l = frame_spikes[self.descending_map.get("P9_oDN1_left", 0)] / dt
        odn_r = frame_spikes[self.descending_map.get("P9_oDN1_right", 0)] / dt
        odn_mean = (odn_l + odn_r) / 2.0
        self.smooth_odn += alpha * (odn_mean - self.smooth_odn)
        forward_thrust = float(np.clip(self.smooth_odn / config.MAX_FORWARD_THRUST_HZ, 0.0, 1.0))

        # 2. Steering Yaw Torque: DNa02 (right - left differential)
        dna_l = frame_spikes[self.descending_map.get("DNa02_left", 0)] / dt
        dna_r = frame_spikes[self.descending_map.get("DNa02_right", 0)] / dt
        self.smooth_dna_l += alpha * (dna_l - self.smooth_dna_l)
        self.smooth_dna_r += alpha * (dna_r - self.smooth_dna_r)
        # If DNa02_right fires faster -> turn right (positive yaw)
        diff_torque = (self.smooth_dna_r - self.smooth_dna_l) / config.MAX_YAW_TORQUE_HZ
        yaw_torque = float(np.clip(diff_torque, -1.0, 1.0))

        # 3. Proboscis Feeding Motor Neurons: MN9 (left + right)
        mn9_l = frame_spikes[self.mn9_map.get("MN9_left", 0)] / dt
        mn9_r = frame_spikes[self.mn9_map.get("MN9_right", 0)] / dt
        mn9_mean = (mn9_l + mn9_r) / 2.0
        self.smooth_mn9 += alpha * (mn9_mean - self.smooth_mn9)

        proboscis_ext = float(np.clip(self.smooth_mn9 / config.MN9_MAX_EXTENSION_HZ, 0.0, 1.0))
        is_eating = bool(self.smooth_mn9 >= config.MN9_EATING_THRESHOLD_HZ)

        # 4. Backward Walking / Escape: MDN moonwalker neurons
        mdn_keys = ["MDN_1", "MDN_2", "MDN_3", "MDN_4"]
        mdn_rates = [frame_spikes[self.descending_map[k]] / dt for k in mdn_keys if k in self.descending_map]
        mdn_mean = float(np.mean(mdn_rates)) if mdn_rates else 0.0
        self.smooth_mdn += alpha * (mdn_mean - self.smooth_mdn)
        is_escaping = bool(self.smooth_mdn > 12.0)

        # Biological Coordination Rule:
        # If the fly is actively feeding (eating), forward walking and turning are paused
        if is_eating:
            forward_thrust = 0.0
            yaw_torque = 0.0

        return {
            "type": "motor_frame",
            "locomotion": {
                "forward_thrust": round(forward_thrust, 3),
                "yaw_torque": round(yaw_torque, 3),
                "is_walking": bool(forward_thrust > 0.05),
                "is_escaping": is_escaping,
            },
            "feeding": {
                "proboscis_extension": round(proboscis_ext, 3),
                "is_eating": is_eating,
                "mn9_rate_hz": round(float(self.smooth_mn9), 1),
            },
            "telemetry": {
                "p9_odn1_hz": round(float(self.smooth_odn), 1),
                "dna02_left_hz": round(float(self.smooth_dna_l), 1),
                "dna02_right_hz": round(float(self.smooth_dna_r), 1),
            },
        }
