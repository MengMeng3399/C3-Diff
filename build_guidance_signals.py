import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from dgl.data.utils import load_graphs


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build CF-Diff style 1/2/3-hop guidance signals from a homogeneous DGL train graph."
    )
    parser.add_argument(
        "--graph-path",
        type=str,
        required=True,
        help="Path to graph.bin saved by dgl.data.utils.save_graphs",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Directory to save guidance files",
    )
    parser.add_argument(
        "--num-mashups",
        type=int,
        required=True,
        help="Number of mashup nodes. Mashup ids are [0, num_mashups-1]",
    )
    parser.add_argument(
        "--train-graph-index",
        type=int,
        default=0,
        help="Index of train_g in graph.bin. Default: 0",
    )
    parser.add_argument(
        "--save-dense",
        action="store_true",
        help="Whether to additionally save dense normalized hop vectors",
    )
    return parser.parse_args()


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def pack_hop(
    counter: Counter,
    node_type: str,
    num_mashups: int,
) -> Dict[str, torch.Tensor]:
    """
    counter keys are LOCAL ids within their own type space:
      - mashup local ids: [0, num_mashups-1]
      - api local ids: [0, num_apis-1]

    We save both:
      - local_ids: index inside type-specific embedding table
      - global_ids: original node ids in the graph
      - counts
      - weights
    """
    if node_type not in {"mashup", "api"}:
        raise ValueError(f"node_type must be 'mashup' or 'api', got {node_type}")

    if len(counter) == 0:
        return {
            "local_ids": torch.empty(0, dtype=torch.long),
            "global_ids": torch.empty(0, dtype=torch.long),
            "counts": torch.empty(0, dtype=torch.long),
            "weights": torch.empty(0, dtype=torch.float),
        }

    local_ids = sorted(counter.keys())
    counts = torch.tensor([counter[i] for i in local_ids], dtype=torch.long)
    weights = counts.float() / counts.sum().float()

    local_ids_tensor = torch.tensor(local_ids, dtype=torch.long)
    if node_type == "mashup":
        global_ids_tensor = local_ids_tensor.clone()
    else:
        global_ids_tensor = local_ids_tensor + num_mashups

    return {
        "local_ids": local_ids_tensor,
        "global_ids": global_ids_tensor,
        "counts": counts,
        "weights": weights,
    }


def extract_train_pairs_from_homo_graph(
    g,
    num_mashups: int,
) -> Tuple[List[Tuple[int, int]], int]:
    """
    Extract unique (mashup_local_id, api_local_id) pairs from a homogeneous DGLGraph.

    Node id convention:
      mashup global ids: [0, num_mashups-1]
      api global ids:    [num_mashups, num_nodes-1]

    Return:
      pairs: list of (mashup_local_id, api_local_id)
      num_apis
    """
    num_nodes = g.num_nodes()
    if num_mashups <= 0 or num_mashups >= num_nodes:
        raise ValueError(
            f"Invalid num_mashups={num_mashups}, total graph nodes={num_nodes}"
        )

    num_apis = num_nodes - num_mashups
    api_offset = num_mashups

    src, dst = g.edges()
    src = src.tolist()
    dst = dst.tolist()

    pair_set = set()
    ignored_intra_edges = 0

    for u, v in zip(src, dst):
        # 以下四个为判断部分 判断 u 和 v 是mashup节点还是API节点，判断mashup api的ID到底对不对
        u_is_mashup = 0 <= u < num_mashups   # u_is_mashup = (0 <= u) and (u < num_mashups)
        v_is_mashup = 0 <= v < num_mashups
        u_is_api = api_offset <= u < num_nodes
        v_is_api = api_offset <= v < num_nodes

        # mashup -> api
        if u_is_mashup and v_is_api:
            pair_set.add((u, v - api_offset))

        # api -> mashup
        elif u_is_api and v_is_mashup:
            pair_set.add((v, u - api_offset))

        # ignore mashup-mashup or api-api edges
        else:
            ignored_intra_edges += 1

    pairs = sorted(list(pair_set))
    return pairs, num_apis


def build_bipartite_adjacency(
    train_pairs: List[Tuple[int, int]],
    num_mashups: int,
    num_apis: int,
):
    """
    Build local-id adjacency:

      mashup_to_api[m_local] -> set of api_local
      api_to_mashup[a_local] -> set of mashup_local
    """
    mashup_to_api = [set() for _ in range(num_mashups)]
    api_to_mashup = [set() for _ in range(num_apis)]

    for m, a in train_pairs:
        if not (0 <= m < num_mashups):
            raise ValueError(f"Invalid mashup local id: {m}")
        if not (0 <= a < num_apis):
            raise ValueError(f"Invalid api local id: {a}")

        mashup_to_api[m].add(a)
        api_to_mashup[a].add(m)

    return mashup_to_api, api_to_mashup

# 从当前这一层节点出发，统计下一层每个邻居被连到了多少次。
def expand_counter(
    frontier_nodes: List[int],
    adjacency: List[set],
) -> Counter:
    """
    Count how many incoming links each next-hop node receives
    from the previous-hop frontier.

    count = number of incoming links from previous-hop neighbors.
    """
    counter = Counter()
    for node_id in frontier_nodes:
        for nb in adjacency[node_id]:
            counter[nb] += 1
    return counter


def build_mashup_guidance(
    mashup_to_api: List[set],
    api_to_mashup: List[set],
    num_mashups: int,
):
    """
    For each mashup m (local id):
      hop1 -> api local ids
      hop2 -> mashup local ids
      hop3 -> api local ids

    Exact-hop rules:
      - hop1: direct APIs, binary edges => count=1
      - hop2: exclude self mashup
      - hop3: exclude APIs already in hop1
    """
    all_guidance = []

    for m in range(num_mashups):
        # hop1: direct api neighbors, binary edge => count = 1
        hop1_counter = Counter({a: 1 for a in mashup_to_api[m]})
        hop1_set = set(hop1_counter.keys())

        # hop2: other mashups reached from hop1 apis
        hop2_counter = expand_counter(list(hop1_set), api_to_mashup)
        hop2_counter.pop(m, None)  # remove self
        hop2_set = set(hop2_counter.keys())

        # hop3: apis reached from hop2 mashups
        hop3_counter = expand_counter(list(hop2_set), mashup_to_api)
        for a in hop1_set:
            hop3_counter.pop(a, None)  # exact 3-hop only
        hop3_set = set(hop3_counter.keys())

        # pack
        item = {
            "node_type": "mashup",
            "node_local_id": m,
            "node_global_id": m,
            "hop1": pack_hop(hop1_counter, node_type="api", num_mashups=num_mashups),
            "hop2": pack_hop(hop2_counter, node_type="mashup", num_mashups=num_mashups),
            "hop3": pack_hop(hop3_counter, node_type="api", num_mashups=num_mashups),
        }
        all_guidance.append(item)

    return all_guidance


def build_api_guidance(
    mashup_to_api: List[set],
    api_to_mashup: List[set],
    num_mashups: int,
    num_apis: int,
):
    """
    For each api a (local id):
      hop1 -> mashup local ids
      hop2 -> api local ids
      hop3 -> mashup local ids

    Exact-hop rules:
      - hop1: direct mashups, binary edge => count=1
      - hop2: exclude self api
      - hop3: exclude mashups already in hop1
    """
    all_guidance = []

    for a in range(num_apis):
        api_global_id = num_mashups + a

        # hop1: direct mashup neighbors, binary edge => count = 1
        hop1_counter = Counter({m: 1 for m in api_to_mashup[a]})
        hop1_set = set(hop1_counter.keys())

        # hop2: other apis reached from hop1 mashups
        hop2_counter = expand_counter(list(hop1_set), mashup_to_api)
        hop2_counter.pop(a, None)  # remove self
        hop2_set = set(hop2_counter.keys())

        # hop3: mashups reached from hop2 apis
        hop3_counter = expand_counter(list(hop2_set), api_to_mashup)
        for m in hop1_set:
            hop3_counter.pop(m, None)  # exact 3-hop only
        hop3_set = set(hop3_counter.keys())

        item = {
            "node_type": "api",
            "node_local_id": a,
            "node_global_id": api_global_id,
            "hop1": pack_hop(hop1_counter, node_type="mashup", num_mashups=num_mashups),
            "hop2": pack_hop(hop2_counter, node_type="api", num_mashups=num_mashups),
            "hop3": pack_hop(hop3_counter, node_type="mashup", num_mashups=num_mashups),
        }
        all_guidance.append(item)

    return all_guidance


def summarize_guidance(guidance: List[Dict], prefix: str) -> Dict[str, float]:
    hop1_sizes = [int(x["hop1"]["local_ids"].numel()) for x in guidance]
    hop2_sizes = [int(x["hop2"]["local_ids"].numel()) for x in guidance]
    hop3_sizes = [int(x["hop3"]["local_ids"].numel()) for x in guidance]

    def mean(xs):
        return float(sum(xs)) / len(xs) if len(xs) > 0 else 0.0

    return {
        f"{prefix}_num_nodes": len(guidance),
        f"{prefix}_hop1_avg_size": mean(hop1_sizes),
        f"{prefix}_hop2_avg_size": mean(hop2_sizes),
        f"{prefix}_hop3_avg_size": mean(hop3_sizes),
        f"{prefix}_hop1_max_size": max(hop1_sizes) if hop1_sizes else 0,
        f"{prefix}_hop2_max_size": max(hop2_sizes) if hop2_sizes else 0,
        f"{prefix}_hop3_max_size": max(hop3_sizes) if hop3_sizes else 0,
    }


def save_dense_optional(
    mashup_guidance: List[Dict],
    api_guidance: List[Dict],
    num_mashups: int,
    num_apis: int,
    out_dir: Path,
):
    """
    Optional dense save for debugging / analysis.

    mashup:
      hop1 -> [num_apis]
      hop2 -> [num_mashups]
      hop3 -> [num_apis]

    api:
      hop1 -> [num_mashups]
      hop2 -> [num_apis]
      hop3 -> [num_mashups]
    """
    mashup_dense = []
    for item in mashup_guidance:
        h1 = torch.zeros(num_apis, dtype=torch.float)
        h2 = torch.zeros(num_mashups, dtype=torch.float)
        h3 = torch.zeros(num_apis, dtype=torch.float)

        if item["hop1"]["local_ids"].numel() > 0:
            h1[item["hop1"]["local_ids"]] = item["hop1"]["weights"]
        if item["hop2"]["local_ids"].numel() > 0:
            h2[item["hop2"]["local_ids"]] = item["hop2"]["weights"]
        if item["hop3"]["local_ids"].numel() > 0:
            h3[item["hop3"]["local_ids"]] = item["hop3"]["weights"]

        mashup_dense.append(
            {
                "node_local_id": item["node_local_id"],
                "node_global_id": item["node_global_id"],
                "hop1": h1,
                "hop2": h2,
                "hop3": h3,
            }
        )

    api_dense = []
    for item in api_guidance:
        h1 = torch.zeros(num_mashups, dtype=torch.float)
        h2 = torch.zeros(num_apis, dtype=torch.float)
        h3 = torch.zeros(num_mashups, dtype=torch.float)

        if item["hop1"]["local_ids"].numel() > 0:
            h1[item["hop1"]["local_ids"]] = item["hop1"]["weights"]
        if item["hop2"]["local_ids"].numel() > 0:
            h2[item["hop2"]["local_ids"]] = item["hop2"]["weights"]
        if item["hop3"]["local_ids"].numel() > 0:
            h3[item["hop3"]["local_ids"]] = item["hop3"]["weights"]

        api_dense.append(
            {
                "node_local_id": item["node_local_id"],
                "node_global_id": item["node_global_id"],
                "hop1": h1,
                "hop2": h2,
                "hop3": h3,
            }
        )

    torch.save(mashup_dense, out_dir / "mashup_guidance_train_dense.pt")
    torch.save(api_dense, out_dir / "api_guidance_train_dense.pt")




def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    graphs, graph_labels = load_graphs(args.graph_path)
    if args.train_graph_index < 0 or args.train_graph_index >= len(graphs):
        raise IndexError(
            f"train_graph_index={args.train_graph_index} out of range, total graphs={len(graphs)}"
        )

    train_g = graphs[args.train_graph_index]

    num_nodes = train_g.num_nodes()
    num_mashups = args.num_mashups
    num_apis = num_nodes - num_mashups

    train_pairs, num_apis_check = extract_train_pairs_from_homo_graph(
        train_g,
        num_mashups=num_mashups,
    )
    assert num_apis == num_apis_check

    mashup_to_api, api_to_mashup = build_bipartite_adjacency(
        train_pairs=train_pairs,
        num_mashups=num_mashups,
        num_apis=num_apis,
    )

    mashup_guidance = build_mashup_guidance(
        mashup_to_api=mashup_to_api,
        api_to_mashup=api_to_mashup,
        num_mashups=num_mashups,
    )
    api_guidance = build_api_guidance(
        mashup_to_api=mashup_to_api,
        api_to_mashup=api_to_mashup,
        num_mashups=num_mashups,
        num_apis=num_apis,
    )

    # sparse saves: recommended
    torch.save(mashup_guidance, out_dir / "mashup_guidance_train.pt")
    torch.save(api_guidance, out_dir / "api_guidance_train.pt")

    combined = {
        "mashup_guidance": mashup_guidance,
        "api_guidance": api_guidance,
        "num_mashups": num_mashups,
        "num_apis": num_apis,
        "num_nodes": num_nodes,
        "num_train_pairs": len(train_pairs),
        "graph_path": args.graph_path,
        "train_graph_index": args.train_graph_index,
        "graph_type": "homogeneous_dgl_graph",
        "node_id_rule": {
            "mashup_global_ids": [0, num_mashups - 1],
            "api_global_ids": [num_mashups, num_nodes - 1],
        },
        "signal_style": "CF-Diff style",
        "hop_definition": "exact_hop",
    }
    torch.save(combined, out_dir / "guidance_signals_train.pt")

    if args.save_dense:
        save_dense_optional(
            mashup_guidance=mashup_guidance,
            api_guidance=api_guidance,
            num_mashups=num_mashups,
            num_apis=num_apis,
            out_dir=out_dir,
        )

    meta = {
        "graph_path": args.graph_path,
        "train_graph_index": args.train_graph_index,
        "num_graphs_in_file": len(graphs),
        "graph_labels_keys": list(graph_labels.keys()) if isinstance(graph_labels, dict) else [],
        "graph_type": "homogeneous_dgl_graph",
        "num_nodes": num_nodes,
        "num_mashups": num_mashups,
        "num_apis": num_apis,
        "num_train_pairs": len(train_pairs),
        "edge_type": "binary",
        "node_id_rule": {
            "mashup_global_ids": [0, num_mashups - 1],
            "api_global_ids": [num_mashups, num_nodes - 1],
        },
        "signal_style": "CF-Diff style",
        "hop_definition": "exact_hop",
        "count_definition": {
            "hop1": "binary direct neighbors, count=1",
            "hop2": "number of incoming links from hop1 frontier",
            "hop3": "number of incoming links from hop2 frontier",
        },
        "weight_definition": "weights = counts / sum(counts) within each hop",
        "notes": [
            "Built from TRAIN graph only",
            "Graph is homogeneous DGLGraph",
            "Node ids are continuous global ids",
            "Sparse files save both local_ids and global_ids",
            "hop2 excludes self node",
            "hop3 excludes nodes already appearing in hop1",
        ],
    }
    meta.update(summarize_guidance(mashup_guidance, "mashup"))
    meta.update(summarize_guidance(api_guidance, "api"))

    with open(out_dir / "guidance_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print("Build guidance signals finished.")
    print(f"Graph path         : {args.graph_path}")
    print(f"Train graph index  : {args.train_graph_index}")
    print(f"Graph type         : homogeneous_dgl_graph")
    print(f"Num nodes          : {num_nodes}")
    print(f"Num mashups        : {num_mashups}")
    print(f"Num apis           : {num_apis}")
    print(f"Num train pairs    : {len(train_pairs)}")
    print(f"Output dir         : {out_dir}")
    print("-" * 80)
    print(f"Sparse mashup file : {out_dir / 'mashup_guidance_train.pt'}")
    print(f"Sparse api file    : {out_dir / 'api_guidance_train.pt'}")
    print(f"Combined file      : {out_dir / 'guidance_signals_train.pt'}")
    if args.save_dense:
        print(f"Dense mashup file  : {out_dir / 'mashup_guidance_train_dense.pt'}")
        print(f"Dense api file     : {out_dir / 'api_guidance_train_dense.pt'}")
    print(f"Meta file          : {out_dir / 'guidance_meta.json'}")
    print("=" * 80)


if __name__ == "__main__":
    main()