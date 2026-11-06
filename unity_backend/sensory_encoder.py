"""
unity_backend/sensory_encoder.py

Translates Unity 3D game observations (compound eye vision angles, collider contact)
into biological firing frequencies (Hz) for the 5,000-neuron connectome subnetwork.
"""

from typing import Dict, List, Tuple
import torch

from . import config


class SensoryEncoder:
    """Encodes game sensory inputs into neuron population firing rates."""

    def __init__(self, engine):
        self.engine = engine
        self.device = engine.device
        self.num_neurons = engine.num_neurons

        # Extract group indices from engine metadata
        self.p9_indices = engine.p9_indices.tolist()
        self.sugar_indices = engine.sugar_indices.tolist()
        self.bitter_indices = engine.bitter_indices.tolist()

        # Reusable rate buffer
        self.rates = torch.zeros(self.num_neurons, dtype=torch.float32, device=self.device)

    def encode(self, sensory_data: dict) -> Tuple[torch.Tensor, List[int]]:
        """
        Convert Unity sensory frame into firing rates (Hz).

        Supported payload schema:
        {
            "vision": {
                "target_visible": bool,
                "left_eye_intensity": float,   # 0.0 to 1.0
                "right_eye_intensity": float,  # 0.0 to 1.0
                "target_azimuth_deg": float,   # -180 to +180
                "target_distance": float
            },
            "contact_taste": {
                "touching_sugar": bool,
                "touching_bitter": bool
            }
        }
        """
        self.rates.zero_()
        stimulated_indices = []

        vision = sensory_data.get("vision", {})
        taste = sensory_data.get("contact_taste", {})

        # 1. Visual Target Tracking -> Bilateral P9 Mechanosensory/Visual Inputs
        target_visible = vision.get("target_visible", False)
        if target_visible:
            left_int = float(vision.get("left_eye_intensity", 0.0))
            right_int = float(vision.get("right_eye_intensity", 0.0))

            # Firing rates proportional to visual field excitation with basal target drive
            base_drive = config.BASE_VISUAL_DRIVE_RATE
            max_range = config.MAX_VISUAL_DRIVE_RATE - base_drive
            p9_left_hz = base_drive + max_range * left_int
            p9_right_hz = base_drive + max_range * right_int
        else:
            # Baseline spontaneous exploratory search walk
            p9_left_hz = config.BASE_EXPLORATORY_RATE
            p9_right_hz = config.BASE_EXPLORATORY_RATE

        if len(self.p9_indices) >= 2:
            self.rates[self.p9_indices[0]] = p9_left_hz
            self.rates[self.p9_indices[1]] = p9_right_hz
            if p9_left_hz > 0:
                stimulated_indices.append(self.p9_indices[0])
            if p9_right_hz > 0:
                stimulated_indices.append(self.p9_indices[1])

        # 2. Contact Taste -> Gustatory Receptor Neurons (GRNs)
        if taste.get("touching_sugar", False):
            self.rates[self.sugar_indices] = config.SUGAR_CONTACT_RATE
            stimulated_indices.extend(self.sugar_indices)

        if taste.get("touching_bitter", False):
            self.rates[self.bitter_indices] = config.BITTER_CONTACT_RATE
            stimulated_indices.extend(self.bitter_indices)

        return self.rates, stimulated_indices
