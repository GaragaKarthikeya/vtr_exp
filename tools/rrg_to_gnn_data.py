#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import numpy as np

NODE_TYPES = ["SOURCE", "SINK", "OPIN", "IPIN", "CHANX", "CHANY"]
NODE_TYPE_TO_INDEX = {name: idx for idx, name in enumerate(NODE_TYPES)}
SIDE_TO_INDEX = {"LEFT": 0, "RIGHT": 1, "TOP": 2, "BOTTOM": 3}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert VPR rr_graph + place/route artifacts into GNN-ready arrays"
    )
    parser.add_argument("--rr-graph", type=Path, required=True, help="Path to rr_graph.xml")
    parser.add_argument("--place", type=Path, required=True, help="Path to .place file")
    parser.add_argument("--route", type=Path, default=None, help="Optional .route file for net endpoint mapping")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for generated arrays/json")
    parser.add_argument(
        "--target-net",
        type=str,
        default=None,
        help="Optional net name from .route; if set, SOURCE/SINK nodes of that net get target flags",
    )
    parser.add_argument(
        "--write-pyg",
        action="store_true",
        help="Also export a PyTorch/PyG-ready .pt file",
    )
    parser.add_argument(
        "--pyg-file",
        type=Path,
        default=None,
        help="Optional output .pt path (default: <output-dir>/graph_data.pt)",
    )
    return parser.parse_args()


def parse_place(place_path: Path) -> dict[str, tuple[int, int, int, int]]:
    placements: dict[str, tuple[int, int, int, int]] = {}
    with place_path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue
            parts = raw.split()
            if len(parts) < 6:
                continue
            name = parts[0]
            try:
                x = int(parts[1])
                y = int(parts[2])
                subblk = int(parts[3])
                layer = int(parts[4])
            except ValueError:
                continue
            block_number_token = parts[5].lstrip("#")
            try:
                block_number = int(block_number_token)
            except ValueError:
                block_number = -1
            placements[name] = (x, y, subblk, layer)
    return placements


def parse_route_endpoints(route_path: Path | None) -> dict[str, dict[str, list[int]]]:
    if route_path is None or not route_path.is_file():
        return {}

    net_to_nodes: dict[str, dict[str, list[int]]] = defaultdict(lambda: {"sources": [], "sinks": []})
    current_net = None

    net_header = re.compile(r"^Net\s+\d+\s+\((.*)\)$")
    source_line = re.compile(r"Node:\s*(\d+)\s+SOURCE")
    sink_line = re.compile(r"Node:\s*(\d+)\s+SINK")

    with route_path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.rstrip("\n")
            net_match = net_header.match(text.strip())
            if net_match:
                current_net = net_match.group(1)
                _ = net_to_nodes[current_net]
                continue
            if current_net is None:
                continue

            src_match = source_line.search(text)
            if src_match:
                net_to_nodes[current_net]["sources"].append(int(src_match.group(1)))
                continue

            sink_match = sink_line.search(text)
            if sink_match:
                net_to_nodes[current_net]["sinks"].append(int(sink_match.group(1)))

    for net_name, bucket in net_to_nodes.items():
        bucket["sources"] = sorted(set(bucket["sources"]))
        bucket["sinks"] = sorted(set(bucket["sinks"]))

    return dict(net_to_nodes)


def parse_rr_graph(rr_graph_path: Path) -> tuple[list[dict[str, float | int | str]], list[tuple[int, int, int]], dict[str, int]]:
    tree = ET.parse(rr_graph_path)
    root = tree.getroot()

    channel = root.find("channels/channel")
    max_x = int(channel.attrib.get("x_max", "1")) if channel is not None else 1
    max_y = int(channel.attrib.get("y_max", "1")) if channel is not None else 1

    rr_nodes = root.find("rr_nodes")
    rr_edges = root.find("rr_edges")

    if rr_nodes is None or rr_edges is None:
        raise ValueError("rr_graph.xml does not contain rr_nodes/rr_edges blocks")

    nodes: list[dict[str, float | int | str]] = []
    for node in rr_nodes.findall("node"):
        node_id = int(node.attrib["id"])
        node_type = node.attrib.get("type", "")
        capacity = int(node.attrib.get("capacity", "1"))

        loc = node.find("loc")
        timing = node.find("timing")

        xlow = int(loc.attrib.get("xlow", "0")) if loc is not None else 0
        xhigh = int(loc.attrib.get("xhigh", str(xlow))) if loc is not None else xlow
        ylow = int(loc.attrib.get("ylow", "0")) if loc is not None else 0
        yhigh = int(loc.attrib.get("yhigh", str(ylow))) if loc is not None else ylow
        ptc = int(loc.attrib.get("ptc", "0")) if loc is not None else 0
        layer = int(loc.attrib.get("layer", "0")) if loc is not None else 0
        side = loc.attrib.get("side", "") if loc is not None else ""

        r_val = float(timing.attrib.get("R", "0")) if timing is not None else 0.0
        c_val = float(timing.attrib.get("C", "0")) if timing is not None else 0.0

        x_center = (xlow + xhigh) / 2.0
        y_center = (ylow + yhigh) / 2.0

        nodes.append(
            {
                "id": node_id,
                "type": node_type,
                "capacity": capacity,
                "xlow": xlow,
                "xhigh": xhigh,
                "ylow": ylow,
                "yhigh": yhigh,
                "x_center": x_center,
                "y_center": y_center,
                "x_norm": x_center / max(1, max_x),
                "y_norm": y_center / max(1, max_y),
                "ptc": ptc,
                "layer": layer,
                "side": side,
                "R": r_val,
                "C": c_val,
            }
        )

    nodes.sort(key=lambda n: int(n["id"]))

    id_to_index = {int(n["id"]): idx for idx, n in enumerate(nodes)}

    edges: list[tuple[int, int, int]] = []
    for edge in rr_edges.findall("edge"):
        src_node = int(edge.attrib["src_node"])
        sink_node = int(edge.attrib["sink_node"])
        switch_id = int(edge.attrib.get("switch_id", "0"))
        if src_node not in id_to_index or sink_node not in id_to_index:
            continue
        edges.append((id_to_index[src_node], id_to_index[sink_node], switch_id))

    dims = {"max_x": max_x, "max_y": max_y}
    return nodes, edges, dims


def build_features(
    nodes: list[dict[str, float | int | str]],
    placed_tiles: set[tuple[int, int]],
    target_sources: set[int],
    target_sinks: set[int],
) -> np.ndarray:
    one_hot_len = len(NODE_TYPES)
    side_len = 4
    # [one_hot(6), x_norm, y_norm, capacity, R, C, ptc_norm, layer, side_onehot(4), placed_flag, target_source, target_sink]
    feature_size = one_hot_len + 1 + 1 + 1 + 1 + 1 + 1 + 1 + side_len + 1 + 1 + 1

    x = np.zeros((len(nodes), feature_size), dtype=np.float32)

    max_ptc = max((int(n["ptc"]) for n in nodes), default=1)

    for i, n in enumerate(nodes):
        node_type = str(n["type"])
        if node_type in NODE_TYPE_TO_INDEX:
            x[i, NODE_TYPE_TO_INDEX[node_type]] = 1.0

        offset = one_hot_len
        x[i, offset + 0] = float(n["x_norm"])
        x[i, offset + 1] = float(n["y_norm"])
        x[i, offset + 2] = float(n["capacity"])
        x[i, offset + 3] = float(n["R"])
        x[i, offset + 4] = float(n["C"])
        x[i, offset + 5] = float(n["ptc"]) / max(1.0, float(max_ptc))
        x[i, offset + 6] = float(n["layer"])

        side = str(n["side"])
        side_base = offset + 7
        if side in SIDE_TO_INDEX:
            x[i, side_base + SIDE_TO_INDEX[side]] = 1.0

        x_coord = int(float(n["x_center"]))
        y_coord = int(float(n["y_center"]))
        x[i, side_base + side_len] = 1.0 if (x_coord, y_coord) in placed_tiles else 0.0

        node_id = int(n["id"])
        x[i, side_base + side_len + 1] = 1.0 if node_id in target_sources else 0.0
        x[i, side_base + side_len + 2] = 1.0 if node_id in target_sinks else 0.0

    return x


def export_pyg(
    out_path: Path,
    x: np.ndarray,
    edge_index: np.ndarray,
    edge_switch_id: np.ndarray,
    node_id: np.ndarray,
    metadata: dict,
) -> str:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PyTorch is required for --write-pyg. Install torch, or rerun without --write-pyg."
        ) from exc

    x_t = torch.from_numpy(x)
    edge_index_t = torch.from_numpy(edge_index)
    edge_switch_id_t = torch.from_numpy(edge_switch_id)
    node_id_t = torch.from_numpy(node_id)

    try:
        from torch_geometric.data import Data

        data = Data(
            x=x_t,
            edge_index=edge_index_t,
            edge_switch_id=edge_switch_id_t,
            node_id=node_id_t,
            metadata=metadata,
        )
        torch.save(data, out_path)
        return "pyg_data"
    except ModuleNotFoundError:
        payload = {
            "x": x_t,
            "edge_index": edge_index_t,
            "edge_switch_id": edge_switch_id_t,
            "node_id": node_id_t,
            "metadata": metadata,
        }
        torch.save(payload, out_path)
        return "torch_dict"


def main() -> int:
    args = parse_args()

    rr_graph_path = args.rr_graph.expanduser().resolve()
    place_path = args.place.expanduser().resolve()
    route_path = args.route.expanduser().resolve() if args.route else None
    out_dir = args.output_dir.expanduser().resolve()

    if not rr_graph_path.is_file():
        raise FileNotFoundError(f"rr_graph not found: {rr_graph_path}")
    if not place_path.is_file():
        raise FileNotFoundError(f"place file not found: {place_path}")
    if route_path is not None and not route_path.is_file():
        raise FileNotFoundError(f"route file not found: {route_path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    placements = parse_place(place_path)
    route_endpoints = parse_route_endpoints(route_path)

    target_sources: set[int] = set()
    target_sinks: set[int] = set()
    if args.target_net:
        bucket = route_endpoints.get(args.target_net)
        if bucket is None:
            raise ValueError(f"Target net not found in route file: {args.target_net}")
        target_sources = set(bucket["sources"])
        target_sinks = set(bucket["sinks"])

    nodes, edges, dims = parse_rr_graph(rr_graph_path)

    placed_tiles = {(x, y) for (x, y, _subblk, _layer) in placements.values()}

    x = build_features(nodes, placed_tiles, target_sources, target_sinks)

    if edges:
        edge_index = np.array([(src, dst) for (src, dst, _sw) in edges], dtype=np.int64).T
        edge_switch_id = np.array([sw for (_src, _dst, sw) in edges], dtype=np.int32)
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_switch_id = np.zeros((0,), dtype=np.int32)

    node_id = np.array([int(n["id"]) for n in nodes], dtype=np.int64)
    node_type = np.array([str(n["type"]) for n in nodes], dtype=object)

    np.save(out_dir / "edge_index.npy", edge_index)
    np.save(out_dir / "edge_switch_id.npy", edge_switch_id)
    np.save(out_dir / "node_features.npy", x)
    np.save(out_dir / "node_id.npy", node_id)
    np.save(out_dir / "node_type.npy", node_type)

    with (out_dir / "placements.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "array_size": None,
                "blocks": {
                    name: {"x": v[0], "y": v[1], "subblk": v[2], "layer": v[3]}
                    for name, v in sorted(placements.items())
                },
            },
            f,
            indent=2,
        )

    with (out_dir / "route_endpoints.json").open("w", encoding="utf-8") as f:
        json.dump(route_endpoints, f, indent=2)

    metadata = {
        "node_count": int(len(nodes)),
        "edge_count": int(len(edges)),
        "feature_dim": int(x.shape[1]),
        "node_types": NODE_TYPES,
        "dims": dims,
        "target_net": args.target_net,
        "target_source_count": len(target_sources),
        "target_sink_count": len(target_sinks),
        "rr_graph": str(rr_graph_path),
        "place": str(place_path),
        "route": str(route_path) if route_path else None,
    }
    with (out_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    pyg_kind = None
    pyg_path = None
    if args.write_pyg:
        pyg_path = args.pyg_file.expanduser().resolve() if args.pyg_file else (out_dir / "graph_data.pt")
        pyg_path.parent.mkdir(parents=True, exist_ok=True)
        pyg_kind = export_pyg(
            out_path=pyg_path,
            x=x,
            edge_index=edge_index,
            edge_switch_id=edge_switch_id,
            node_id=node_id,
            metadata=metadata,
        )

    print("Wrote GNN data:")
    print(f"  output_dir: {out_dir}")
    print(f"  nodes: {metadata['node_count']}")
    print(f"  edges: {metadata['edge_count']}")
    print(f"  feature_dim: {metadata['feature_dim']}")
    if args.target_net:
        print(f"  target_net: {args.target_net}")
        print(f"  target_sources: {metadata['target_source_count']}")
        print(f"  target_sinks: {metadata['target_sink_count']}")
    if args.write_pyg:
        print(f"  pyg_export: {pyg_kind}")
        print(f"  pyg_file: {pyg_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
