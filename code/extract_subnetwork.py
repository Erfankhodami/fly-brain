#!/usr/bin/env python3
"""
extract_subnetwork.py

Extracts a biologically coherent 5,000-neuron subnetwork of the Drosophila connectome,
anchored around sensorimotor locomotion/steering and taste/feeding circuits.

Anchors:
- Locomotion inputs: P9 left and right (2 neurons)
- Locomotion / steering descending command neurons:
    P9_oDN1_left, P9_oDN1_right, DNa01_left, DNa01_right,
    DNa02_left, DNa02_right, MDN_1..4 (10 neurons)
- Feeding inputs: 23 Sugar GRNs, 42 Bitter GRNs (65 neurons)
- Feeding motor outputs: MN9 left and right (2 neurons)
- Feeding interneurons: SEZ network from sez_neurons.pickle (369 neurons)

Candidate interneurons are ranked by total synaptic affinity (incident synaptic weight)
to the anchor circuits. Mutual connections among the 5,000 neurons are preserved to form
an induced recurrent subgraph.

Outputs:
- data/subnetwork_5k/subnetwork_meta.json
- data/subnetwork_5k/subnetwork_csr.pt
- data/subnetwork_5k/subnetwork_edges.parquet
"""

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
import torch

# ============================================================================
# Biological Anchor Definitions
# ============================================================================

P9_INPUTS = {
    720575940627652358: "P9_left",
    720575940635872101: "P9_right",
}

DESCENDING_NEURONS = {
    720575940626730883: "P9_oDN1_left",
    720575940620300308: "P9_oDN1_right",
    720575940644438551: "DNa01_left",
    720575940627787609: "DNa01_right",
    720575940604737708: "DNa02_left",
    720575940629327659: "DNa02_right",
    720575940616026939: "MDN_1",
    720575940631082808: "MDN_2",
    720575940640331472: "MDN_3",
    720575940610236514: "MDN_4",
}

SUGAR_GRNS = [
    720575940616885538, 720575940630233916, 720575940639332736, 720575940632889389,
    720575940617000768, 720575940632425919, 720575940637568838, 720575940629176663,
    720575940621502051, 720575940638202345, 720575940612670570, 720575940611875570,
    720575940621754367, 720575940633143833, 720575940613601698, 720575940630797113,
    720575940639198653, 720575940639259967, 720575940624963786, 720575940640649691,
    720575940610788069, 720575940623172843, 720575940628853239
]

BITTER_GRNS = [
    720575940619072513, 720575940646212996, 720575940622298631, 720575940642088333,
    720575940627692048, 720575940617239197, 720575940618682526, 720575940604714528,
    720575940603266592, 720575940604027168, 720575940619197093, 720575940610259370,
    720575940627578156, 720575940629481516, 720575940618887217, 720575940614281266,
    720575940634859188, 720575940645743412, 720575940637742911, 720575940617094208,
    720575940629416318, 720575940630195909, 720575940615641798, 720575940638312262,
    720575940624310345, 720575940621778381, 720575940619659861, 720575940629146711,
    720575940625750105, 720575940610483162, 720575940610481370, 720575940602353632,
    720575940610773090, 720575940617433830, 720575940628962407, 720575940626287336,
    720575940623183083, 720575940618025199, 720575940619028208, 720575940621864060,
    720575940613061118, 720575940621008895
]

MOTOR_NEURONS = {
    720575940660219265: "MN9_left",
    720575940618238523: "MN9_right",
}

EXTRA_COMMAND_NEURONS = {
    720575940622838154: "Giant_Fiber_1",
    720575940632499757: "Giant_Fiber_2",
    720575940616185531: "aDN1_right",
    720575940624319124: "aDN1_left",
}


def load_sez_neurons(sez_path):
    """Load SEZ neuron IDs and cell types from pickle."""
    with open(sez_path, "rb") as f:
        sez_dict = pickle.load(f)
    sez_ids = set()
    id2type = {}
    for cell_type, ids in sez_dict.items():
        if isinstance(ids, (list, set, tuple)):
            for nid in ids:
                sez_ids.add(int(nid))
                id2type[int(nid)] = cell_type
        elif isinstance(ids, int):
            sez_ids.add(int(ids))
            id2type[int(ids)] = cell_type
    return sez_ids, id2type


def collect_seeds(sez_path):
    """Collect all biological anchor seed neurons with metadata."""
    sez_ids, sez_types = load_sez_neurons(sez_path)

    seed_dict = {}  # flywire_id -> {"group": ..., "label": ...}

    # P9 inputs
    for nid, label in P9_INPUTS.items():
        seed_dict[nid] = {"group": "p9_input", "label": label}

    # Walking / steering descending neurons
    for nid, label in DESCENDING_NEURONS.items():
        seed_dict[nid] = {"group": "descending", "label": label}

    # Sugar GRNs
    for i, nid in enumerate(SUGAR_GRNS):
        seed_dict[nid] = {"group": "sugar_grn", "label": f"sugar_GRN_{i+1}"}

    # Bitter GRNs
    for i, nid in enumerate(BITTER_GRNS):
        seed_dict[nid] = {"group": "bitter_grn", "label": f"bitter_GRN_{i+1}"}

    # Motor neurons
    for nid, label in MOTOR_NEURONS.items():
        seed_dict[nid] = {"group": "motor_mn9", "label": label}

    # Extra command neurons
    for nid, label in EXTRA_COMMAND_NEURONS.items():
        if nid not in seed_dict:
            seed_dict[nid] = {"group": "command_extra", "label": label}

    # SEZ interneurons
    for nid in sez_ids:
        if nid not in seed_dict:
            ctype = sez_types.get(nid, "SEZ_interneuron")
            seed_dict[nid] = {"group": "sez_interneuron", "label": f"SEZ_{ctype}_{nid}"}

    return seed_dict


def extract_subnetwork(
    conn_path: str,
    comp_path: str,
    sez_path: str,
    output_dir: str,
    target_neurons: int = 5000,
    min_weight: int = 1,
):
    start_time = time.time()
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Drosophila Subnetwork Extractor")
    print(f"Target neurons: {target_neurons}")
    print(f"Output directory: {out_path.resolve()}")
    print("=" * 70)

    # 1. Collect biological anchor seed neurons
    print("\n[1/6] Collecting biological anchor seeds...")
    seed_dict = collect_seeds(sez_path)
    seed_ids = set(seed_dict.keys())
    print(f"  Anchor seeds collected: {len(seed_ids)} neurons")
    for grp in ["p9_input", "descending", "sugar_grn", "bitter_grn", "motor_mn9", "command_extra", "sez_interneuron"]:
        cnt = sum(1 for v in seed_dict.values() if v["group"] == grp)
        print(f"    - {grp}: {cnt} neurons")

    # 2. Read connectivity parquet table
    print("\n[2/6] Reading connectivity table from parquet...")
    t_read_start = time.time()
    table = pq.read_table(
        conn_path,
        columns=[
            "Presynaptic_ID",
            "Postsynaptic_ID",
            "Presynaptic_Index",
            "Postsynaptic_Index",
            "Connectivity",
            "Excitatory",
            "Excitatory x Connectivity",
        ],
    )
    print(f"  Loaded {table.num_rows:,} total synapses in {time.time() - t_read_start:.2f}s")

    # 3. Score candidate partner neurons by affinity with anchor seeds
    print("\n[3/6] Scoring candidate partner neurons by synaptic affinity...")
    pre_arr = table["Presynaptic_ID"]
    post_arr = table["Postsynaptic_ID"]
    conn_arr = table["Connectivity"]

    # Filter edges incident to any seed neuron
    seed_pa_arr = pa.array(list(seed_ids), type=pa.int64())
    pre_is_seed = pc.is_in(pre_arr, value_set=seed_pa_arr)
    post_is_seed = pc.is_in(post_arr, value_set=seed_pa_arr)
    incident_mask = pc.or_(pre_is_seed, post_is_seed)

    t_incident = table.filter(incident_mask)
    print(f"  Found {t_incident.num_rows:,} edges incident to the {len(seed_ids)} anchor seeds")

    # Accumulate affinity scores for non-seed neurons
    inc_pre = t_incident["Presynaptic_ID"].to_numpy(zero_copy_only=False)
    inc_post = t_incident["Postsynaptic_ID"].to_numpy(zero_copy_only=False)
    inc_conn = t_incident["Connectivity"].to_numpy(zero_copy_only=False)

    affinity_scores = defaultdict(int)
    for pre, post, weight in zip(inc_pre, inc_post, inc_conn):
        if pre not in seed_ids:
            affinity_scores[pre] += weight
        if post not in seed_ids:
            affinity_scores[post] += weight

    print(f"  Identified {len(affinity_scores):,} 1-hop candidate partner neurons")

    # Sort candidates by descending affinity score
    sorted_candidates = sorted(affinity_scores.items(), key=lambda x: x[1], reverse=True)

    needed = target_neurons - len(seed_ids)
    if len(sorted_candidates) < needed:
        raise RuntimeError(
            f"Insufficient candidate neighbors ({len(sorted_candidates)}) for target {target_neurons}"
        )

    selected_partners = [nid for nid, _ in sorted_candidates[:needed]]
    print(f"  Selected top {len(selected_partners):,} candidate interneurons (affinity min: {sorted_candidates[needed-1][1]}, max: {sorted_candidates[0][1]})")

    # Combine seeds and selected partners into the final 5,000 neurons
    # Group and order them systematically:
    # 0: p9_input (2)
    # 1: sugar_grn (23)
    # 2: bitter_grn (42)
    # 3: descending (10)
    # 4: motor_mn9 (2)
    # 5: command_extra (4)
    # 6: sez_interneuron
    # 7: selected_partners (candidate interneurons)
    ordered_neurons = []
    group_order = [
        "p9_input",
        "sugar_grn",
        "bitter_grn",
        "descending",
        "motor_mn9",
        "command_extra",
        "sez_interneuron",
    ]

    group_indices = {}
    current_idx = 0

    for grp in group_order:
        grp_ids = [nid for nid, v in seed_dict.items() if v["group"] == grp]
        # Sort for determinism
        grp_ids.sort()
        start = current_idx
        end = current_idx + len(grp_ids)
        group_indices[grp] = {"start": start, "end": end, "count": len(grp_ids)}
        for nid in grp_ids:
            ordered_neurons.append((nid, seed_dict[nid]["group"], seed_dict[nid]["label"]))
            current_idx += 1

    # Add partner interneurons
    partner_start = current_idx
    partner_end = current_idx + len(selected_partners)
    group_indices["partner_interneurons"] = {
        "start": partner_start,
        "end": partner_end,
        "count": len(selected_partners),
    }
    for i, nid in enumerate(selected_partners):
        ordered_neurons.append((nid, "partner_interneuron", f"partner_{i+1}"))
        current_idx += 1

    assert len(ordered_neurons) == target_neurons, f"Expected {target_neurons}, got {len(ordered_neurons)}"
    print(f"\n[4/6] Final subnetwork structured: {len(ordered_neurons)} neurons exactly.")

    # Build mapping dictionaries
    flyid2local = {nid: idx for idx, (nid, _, _) in enumerate(ordered_neurons)}
    local2flyid = [nid for nid, _, _ in ordered_neurons]

    # 4. Extract induced subgraph of connections among selected neurons
    print("\n[5/6] Extracting induced recurrent synaptic edges...")
    all_selected_set = set(local2flyid)
    all_selected_pa = pa.array(local2flyid, type=pa.int64())

    sub_pre_in = pc.is_in(table["Presynaptic_ID"], value_set=all_selected_pa)
    sub_post_in = pc.is_in(table["Postsynaptic_ID"], value_set=all_selected_pa)
    sub_mask = pc.and_(sub_pre_in, sub_post_in)

    if min_weight > 1:
        weight_mask = pc.greater_equal(pc.abs(table["Excitatory x Connectivity"]), min_weight)
        sub_mask = pc.and_(sub_mask, weight_mask)

    sub_table = table.filter(sub_mask)
    num_edges = sub_table.num_rows
    print(f"  Induced subgraph contains {num_edges:,} directed synaptic edges")
    density = num_edges / (target_neurons * target_neurons)
    print(f"  Connection density: {density:.4%}")

    # Convert to pandas for local index mapping
    sub_df = sub_table.to_pandas()
    sub_df["pre_local_idx"] = sub_df["Presynaptic_ID"].map(flyid2local).astype(np.int32)
    sub_df["post_local_idx"] = sub_df["Postsynaptic_ID"].map(flyid2local).astype(np.int32)

    # 5. Build PyTorch CSR Sparse Matrix
    print("\n[6/6] Building PyTorch CSR sparse tensor and exporting...")
    # Rows = postsynaptic, Cols = presynaptic
    rows = sub_df["post_local_idx"].values
    cols = sub_df["pre_local_idx"].values
    vals = sub_df["Excitatory x Connectivity"].values.astype(np.float32)

    coo_indices = torch.tensor(np.stack([rows, cols]), dtype=torch.int64)
    coo_values = torch.tensor(vals, dtype=torch.float32)
    coo_tensor = torch.sparse_coo_tensor(
        coo_indices, coo_values, (target_neurons, target_neurons)
    ).coalesce()
    csr_tensor = coo_tensor.to_sparse_csr()

    # Save CSR tensor
    csr_path = out_path / "subnetwork_csr.pt"
    torch.save(csr_tensor, csr_path)
    print(f"  Saved PyTorch CSR tensor -> {csr_path} ({os.path.getsize(csr_path) / 1024 / 1024:.2f} MB)")

    # Save edges table as parquet
    edges_pq_path = out_path / "subnetwork_edges.parquet"
    sub_df.to_parquet(edges_pq_path, compression="brotli", index=False)
    print(f"  Saved edges table -> {edges_pq_path} ({os.path.getsize(edges_pq_path) / 1024 / 1024:.2f} MB)")

    # Prepare metadata
    neuron_catalog = [
        {
            "local_index": int(idx),
            "flywire_id": int(nid),
            "group": str(grp),
            "label": str(lbl),
        }
        for idx, (nid, grp, lbl) in enumerate(ordered_neurons)
    ]

    meta = {
        "target_neurons": int(target_neurons),
        "num_edges": int(num_edges),
        "density": float(density),
        "min_weight": int(min_weight),
        "groups": {k: {gk: int(gv) for gk, gv in v.items()} for k, v in group_indices.items()},
        "p9_input_indices": [int(flyid2local[nid]) for nid in P9_INPUTS.keys()],
        "sugar_grn_indices": [int(flyid2local[nid]) for nid in SUGAR_GRNS],
        "bitter_grn_indices": [int(flyid2local[nid]) for nid in BITTER_GRNS],
        "descending_indices": {lbl: int(flyid2local[nid]) for nid, lbl in DESCENDING_NEURONS.items()},
        "mn9_motor_indices": {lbl: int(flyid2local[nid]) for nid, lbl in MOTOR_NEURONS.items()},
        "neurons": neuron_catalog,
        "flyid2local": {str(k): int(v) for k, v in flyid2local.items()},
        "local2flyid": [int(nid) for nid in local2flyid],
    }

    meta_path = out_path / "subnetwork_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved metadata -> {meta_path} ({os.path.getsize(meta_path) / 1024 / 1024:.2f} MB)")

    total_elapsed = time.time() - start_time
    print("=" * 70)
    print(f"Subnetwork extraction completed successfully in {total_elapsed:.2f}s!")
    print(f"Total neurons: {target_neurons:,}")
    print(f"Total synapses: {num_edges:,}")
    print("=" * 70)
    return meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract 5,000-neuron Drosophila subnetwork.")
    parser.add_argument("--target-neurons", type=int, default=5000, help="Target neuron count (default: 5000)")
    parser.add_argument("--output-dir", type=str, default="data/subnetwork_5k", help="Output directory")
    parser.add_argument("--conn-path", type=str, default="data/2025_Connectivity_783.parquet", help="Path to connectivity parquet")
    parser.add_argument("--comp-path", type=str, default="data/2025_Completeness_783.csv", help="Path to completeness CSV")
    parser.add_argument("--sez-path", type=str, default="data/sez_neurons.pickle", help="Path to SEZ neurons pickle")
    parser.add_argument("--min-weight", type=int, default=1, help="Minimum absolute synaptic weight")

    args = parser.parse_args()

    extract_subnetwork(
        conn_path=args.conn_path,
        comp_path=args.comp_path,
        sez_path=args.sez_path,
        output_dir=args.output_dir,
        target_neurons=args.target_neurons,
        min_weight=args.min_weight,
    )
