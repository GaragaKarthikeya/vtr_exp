#!/usr/bin/env python3

import os
import sys
import re
import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np

# Node types for one-hot encoding
NODE_TYPES = ["INPUT", "OUTPUT", "LUT", "LATCH", "SUBCKT"]
TYPE_TO_IDX = {t: i for i, t in enumerate(NODE_TYPES)}

def parse_blif(blif_path: Path):
    """
    Parses a BLIF file and returns a graph structure.
    Nodes are blocks (gates, latches, I/Os).
    Edges are nets.
    """
    nodes = []
    # net_name -> list of (node_id, port_type)
    # port_type: 0 for driver (source), 1 for sink
    nets = defaultdict(list)
    
    current_node_id = 0
    
    with blif_path.open('r') as f:
        content = f.read()
        
    # Handle line continuations
    content = content.replace('\\\n', ' ')
    lines = content.splitlines()
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
            
        parts = line.split()
        cmd = parts[0]
        
        if cmd == '.inputs':
            for net in parts[1:]:
                nodes.append({
                    'id': current_node_id,
                    'type': 'INPUT',
                    'name': net
                })
                nets[net].append((current_node_id, 0)) # Input is a source
                current_node_id += 1
                
        elif cmd == '.outputs':
            for net in parts[1:]:
                nodes.append({
                    'id': current_node_id,
                    'type': 'OUTPUT',
                    'name': net
                })
                nets[net].append((current_node_id, 1)) # Output is a sink
                current_node_id += 1
                
        elif cmd == '.names':
            # Last name is the output (driver), others are inputs (sinks)
            net_names = parts[1:]
            if not net_names:
                continue
            
            output_net = net_names[-1]
            input_nets = net_names[:-1]
            
            nodes.append({
                'id': current_node_id,
                'type': 'LUT',
                'name': output_net,
                'num_inputs': len(input_nets)
            })
            
            nets[output_net].append((current_node_id, 0)) # Driver
            for inet in input_nets:
                nets[inet].append((current_node_id, 1)) # Sink
            current_node_id += 1
            
        elif cmd == '.latch':
            # .latch <input> <output> [<type> <control> <init-val>]
            if len(parts) < 3:
                continue
            input_net = parts[1]
            output_net = parts[2]
            
            nodes.append({
                'id': current_node_id,
                'type': 'LATCH',
                'name': f"latch_{output_net}"
            })
            
            nets[input_net].append((current_node_id, 1)) # Sink
            nets[output_net].append((current_node_id, 0)) # Driver
            current_node_id += 1
            
        elif cmd == '.subckt':
            # .subckt <model-name> <formal1>=<actual1> <formal2>=<actual2> ...
            if len(parts) < 2:
                continue
            model_name = parts[1]
            connections = parts[2:]
            
            node_info = {
                'id': current_node_id,
                'type': 'SUBCKT',
                'model': model_name,
                'name': f"{model_name}_{current_node_id}"
            }
            nodes.append(node_info)
            
            for conn in connections:
                if '=' not in conn:
                    continue
                formal, actual = conn.split('=', 1)
                # We don't know if formal is input or output without reading the library/model definition.
                # Heuristic: VTR/Yosys typically uses 'a', 'b', 'clk' for inputs and 'out', 'q' for outputs.
                # Common patterns:
                # multiply: a, b (in), out (out)
                # adder: a, b, cin (in), out, cout (out)
                # single_port_ram: addr, data, we, clk (in), out (out)
                is_output = False
                if any(x in formal.lower() for x in ['out', 'q', 'cout']):
                    is_output = True
                
                if is_output:
                    nets[actual].append((current_node_id, 0))
                else:
                    nets[actual].append((current_node_id, 1))
            
            current_node_id += 1

    # Build Edge Index
    edge_index = []
    for net_name, participants in nets.items():
        if net_name in ['vcc', 'gnd', 'unconn']:
            continue
            
        drivers = [p[0] for p in participants if p[1] == 0]
        sinks = [p[0] for p in participants if p[1] == 1]
        
        for d in drivers:
            for s in sinks:
                if d != s:
                    edge_index.append((d, s))
                    
    return nodes, edge_index

def build_features(nodes):
    num_nodes = len(nodes)
    num_features = len(NODE_TYPES) + 1 # One-hot + num_inputs
    
    x = np.zeros((num_nodes, num_features), dtype=np.float32)
    
    for i, node in enumerate(nodes):
        # Type One-Hot
        type_idx = TYPE_TO_IDX.get(node['type'], 0)
        x[i, type_idx] = 1.0
        
        # Num Inputs (normalized roughly)
        if node['type'] == 'LUT':
            x[i, -1] = node.get('num_inputs', 0) / 6.0
        elif node['type'] == 'SUBCKT':
            x[i, -1] = 1.0 # Treat subckt as "complex"
            
    return x

def export_to_pyg(nodes, edge_index, output_path):
    import torch
    from torch_geometric.data import Data
    
    x = torch.from_numpy(build_features(nodes))
    if edge_index:
        edge_index_t = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    else:
        edge_index_t = torch.zeros((2, 0), dtype=torch.long)
        
    data = Data(x=x, edge_index=edge_index_t)
    torch.save(data, output_path)
    print(f"Exported PyG data to {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("blif", type=Path)
    parser.add_argument("--output", type=Path, default=Path("circuit_graph.pt"))
    args = parser.parse_args()
    
    if not args.blif.exists():
        print(f"Error: {args.blif} not found")
        sys.exit(1)
        
    nodes, edge_index = parse_blif(args.blif)
    print(f"Parsed {len(nodes)} nodes and {len(edge_index)} edges from {args.blif}")
    
    try:
        import torch
        import torch_geometric
        export_to_pyg(nodes, edge_index, args.output)
    except ImportError:
        print("Warning: torch or torch_geometric not found. Saving as numpy arrays instead.")
        x = build_features(nodes)
        edge_index_np = np.array(edge_index)
        np.save(args.output.with_suffix('.x.npy'), x)
        np.save(args.output.with_suffix('.edge_index.npy'), edge_index_np)
        print(f"Saved to {args.output.with_suffix('.x.npy')} and {args.output.with_suffix('.edge_index.npy')}")

if __name__ == "__main__":
    main()
