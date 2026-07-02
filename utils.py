
from __future__ import annotations

import ast
import json
import os
import random
from typing import Dict, Iterable, List, Tuple

import dgl
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer


# ================================================================
# 0. BASIC HELPERS
# ================================================================

def matrix_to_graph(matrix: np.ndarray, use_weight: bool = False) -> dgl.DGLGraph:
    """
    Convert a dense matrix into a homogeneous bipartite graph.

    Nodes are indexed as:
        - [0, rows - 1] for the row side,
        - [rows, rows + cols - 1] for the column side.

    Parameters
    ----------
    matrix : np.ndarray
        Dense matrix of shape (num_rows, num_cols).
    use_weight : bool, optional
        If True, edge weights are stored in `g.edata["w"]`.

    Returns
    -------
    dgl.DGLGraph
        Constructed bipartite graph.
    """
    nonzero_indices = np.nonzero(matrix)
    src = torch.tensor(nonzero_indices[0], dtype=torch.long)
    dst = torch.tensor(nonzero_indices[1] + matrix.shape[0], dtype=torch.long)
    g = dgl.graph((src, dst))

    if use_weight:
        weights = torch.tensor(
            [matrix[src[i].item(), dst[i].item() - matrix.shape[0]] for i in range(len(src))],
            dtype=torch.float32,
        )
        g.edata["w"] = weights

    return g


def dict_to_graph(offset: int, index_dict: Dict[int, Iterable[int]]) -> dgl.DGLGraph:
    """
    Convert a dictionary representation into a bipartite graph.

    Each key is a source node, and each value list is the set of target indices.
    Target indices are shifted by an offset to avoid id collisions.

    Parameters
    ----------
    offset : int
        Offset added to all target indices.
    index_dict : Dict[int, Iterable[int]]
        Mapping from source node id to a list of target indices.

    Returns
    -------
    dgl.DGLGraph
        Constructed bipartite graph.
    """
    src: List[int] = []
    dst: List[int] = []

    for key, value_list in index_dict.items():
        for j in value_list:
            src.append(int(key))
            dst.append(int(j) + offset)

    src_t = torch.tensor(src, dtype=torch.long)
    dst_t = torch.tensor(dst, dtype=torch.long)
    g = dgl.graph((src_t, dst_t))
    return g


# ================================================================
# 1. SPLIT TRAIN / TEST AND BUILD BIPARTITE GRAPHS
# ================================================================

def split_train_test_by_mashup(
    df: pd.DataFrame,
    test_ratio: float = 0.2,
    seed: int = 42,
    train_path: str = "train.csv",
    test_path: str = "test.csv",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split edges into train/test sets based on Mashup_ID, while ensuring that
    each mashup contributes at least one edge to the test set.

    The split result is saved to two CSV files with no directory prefix.

    Parameters
    ----------
    df : pd.DataFrame
        Original invoke data with at least columns ["Mashup_ID", "Api_ID"].
    test_ratio : float, optional
        Proportion of edges used as test data.
    seed : int, optional
        Random seed for reproducibility.
    train_path : str, optional
        Output CSV file for the train set.
    test_path : str, optional
        Output CSV file for the test set.

    Returns
    -------
    (train_df, test_df) : Tuple[pd.DataFrame, pd.DataFrame]
        DataFrames for train and test edges (same schema as df).
    """
    random.seed(seed)

    df_edges = df[["Mashup_ID", "Api_ID"]].copy()
    edge_list = list(df_edges.itertuples(index=False, name=None))
    random.shuffle(edge_list)

    mashup_ids = sorted(df_edges["Mashup_ID"].unique())
    mashup_to_edges = {m: [] for m in mashup_ids}
    for m, a in edge_list:
        mashup_to_edges[m].append((m, a))

    test_edges: List[Tuple[int, int]] = []
    remaining_edges: List[Tuple[int, int]] = []

    # Ensure at least one test edge per mashup
    for m in mashup_ids:
        edges = mashup_to_edges[m]
        if not edges:
            continue
        random.shuffle(edges)
        test_edges.append(edges[0])
        remaining_edges.extend(edges[1:])

    # Fill up to the desired test size
    total_test_size = int(len(edge_list) * test_ratio)
    extra_needed = max(0, total_test_size - len(test_edges))
    random.shuffle(remaining_edges)

    test_edges += remaining_edges[:extra_needed]
    train_edges = remaining_edges[extra_needed:]

    train_df_base = pd.DataFrame(train_edges, columns=["Mashup_ID", "Api_ID"])
    test_df_base = pd.DataFrame(test_edges, columns=["Mashup_ID", "Api_ID"])

    train_df = df.merge(train_df_base, on=["Mashup_ID", "Api_ID"], how="inner")
    test_df = df.merge(test_df_base, on=["Mashup_ID", "Api_ID"], how="inner")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Train edges saved to: {train_path}   ({len(train_df)} rows)")
    print(f"Test edges saved to:  {test_path}    ({len(test_df)} rows)")
    print(f"Test covers {test_df['Mashup_ID'].nunique()} mashups")

    return train_df, test_df


def build_invoke_graph_from_df(
    invoke_df: pd.DataFrame,
) -> Tuple[dgl.DGLGraph, int, int]:
    """
    Build the original bipartite graph from the full invoke DataFrame.

    Node indexing convention
    ------------------------
    - Mashup nodes: [0, num_mashup - 1]
    - API nodes:    [num_mashup, all_nodes - 1]

    Parameters
    ----------
    invoke_df : pd.DataFrame
        DataFrame with at least columns ["Mashup_ID", "Api_ID"].

    Returns
    -------
    g : dgl.DGLGraph
        Bipartite graph of mashups and APIs.
    all_nodes : int
        Total number of nodes.
    num_mashup : int
        Number of mashup nodes.
    """
    src = invoke_df["Mashup_ID"].to_numpy()
    dst_raw = invoke_df["Api_ID"].to_numpy()

    num_mashup = int(np.max(src)) + 1
    num_api = int(np.max(dst_raw)) + 1
    all_nodes = num_mashup + num_api

    dst = dst_raw + num_mashup
    g = dgl.graph((src, dst), num_nodes=all_nodes)
    return g, all_nodes, num_mashup


def build_subgraphs_from_train_test(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    num_mashup: int,
    all_nodes: int,
    neg_sample_size: int = 5,
    invoke_df: pd.DataFrame | None = None,
) -> Tuple[dgl.DGLGraph, dgl.DGLGraph, dgl.DGLGraph, dgl.DGLGraph, dgl.DGLGraph]:
    """
    Construct positive and negative subgraphs for train/test splits.

    Conventions
    -----------
    - Mashup nodes: [0, num_mashup - 1]
    - API nodes:    [num_mashup, all_nodes - 1]

    Parameters
    ----------
    train_df, test_df : pd.DataFrame
        Train/test edges with columns ["Mashup_ID", "Api_ID"].
    num_mashup : int
        Number of mashup nodes.
    all_nodes : int
        Total number of nodes in the original bipartite graph.
    neg_sample_size : int, optional
        Number of negative samples per positive edge.
    invoke_df : pd.DataFrame, optional
        If provided, negative edges are ensured not to appear
        anywhere in the full invoke data.

    Returns
    -------
    train_g : dgl.DGLGraph
        Graph containing only training positive edges.
    train_pos_g : dgl.DGLGraph
        Same as train_g, kept for clarity.
    train_neg_g : dgl.DGLGraph
        Graph containing negative edges for training.
    test_pos_g : dgl.DGLGraph
        Graph of positive edges for evaluation.
    test_neg_g : dgl.DGLGraph
        Graph of negative edges for evaluation.
    """

    def df_to_edge_list(df: pd.DataFrame) -> List[Tuple[int, int]]:
        m = df["Mashup_ID"].to_numpy()
        a_raw = df["Api_ID"].to_numpy()
        a = a_raw + num_mashup
        return list(zip(m, a))

    train_edges = df_to_edge_list(train_df)
    test_edges = df_to_edge_list(test_df)

    api_nodes = list(range(num_mashup, all_nodes))

    # Positive graphs
    if len(train_edges) > 0:
        train_pos_u, train_pos_v = zip(*train_edges)
    else:
        train_pos_u, train_pos_v = [], []
    test_pos_u, test_pos_v = zip(*test_edges)

    train_pos_u = torch.tensor(train_pos_u, dtype=torch.long)
    train_pos_v = torch.tensor(train_pos_v, dtype=torch.long)
    test_pos_u = torch.tensor(test_pos_u, dtype=torch.long)
    test_pos_v = torch.tensor(test_pos_v, dtype=torch.long)

    train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=all_nodes)
    test_pos_g = dgl.graph((test_pos_u, test_pos_v), num_nodes=all_nodes)

    train_g = train_pos_g

    # Negative sampling
    if invoke_df is not None:
        src_all = invoke_df["Mashup_ID"].to_numpy()
        dst_all_raw = invoke_df["Api_ID"].to_numpy()
        dst_all = dst_all_raw + num_mashup
        all_edge_list = list(zip(src_all, dst_all))
        existing_edge_set = set(all_edge_list)
    else:
        existing_edge_set = set(train_edges) | set(test_edges)

    def negative_sampling(
        pos_edges: List[Tuple[int, int]],
        existing_edges: set[Tuple[int, int]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        neg_u: List[int] = []
        neg_v: List[int] = []
        for m_id, _ in pos_edges:
            for _ in range(neg_sample_size):
                while True:
                    sampled_api = random.choice(api_nodes)
                    if (m_id, sampled_api) not in existing_edges:
                        neg_u.append(m_id)
                        neg_v.append(sampled_api)
                        break
        return (
            torch.tensor(neg_u, dtype=torch.long),
            torch.tensor(neg_v, dtype=torch.long),
        )

    train_neg_u, train_neg_v = negative_sampling(train_edges, existing_edge_set)
    test_neg_u, test_neg_v = negative_sampling(test_edges, existing_edge_set)

    train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=all_nodes)
    test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes=all_nodes)

    return train_g, train_pos_g, train_neg_g, test_pos_g, test_neg_g


def load_data_from_splits(
    args,
    invoke_file: str = "invoke.csv",
    train_file: str = "train.csv",
    test_file: str = "test.csv",
    neg_sample_size: int = 5,
):
    """
    High-level loader given pre-split CSV files.

    Steps
    -----
    1) Load invoke.csv and build the full bipartite graph g.
    2) Load train.csv / test.csv and construct:
       - train_g  (training graph = positive edges)
       - train_pos_g, train_neg_g
       - test_pos_g,  test_neg_g
    3) Return all graphs and basic node counts.

    All file names are assumed to live in `args.data_path` and have no
    extra directory nesting, e.g. `.../invoke.csv`, `.../train.csv`, etc.
    """
    data_dir = args.data_path

    invoke_path = os.path.normpath(os.path.join(data_dir, invoke_file))
    train_path = os.path.normpath(os.path.join(data_dir, train_file))
    test_path = os.path.normpath(os.path.join(data_dir, test_file))

    if not os.path.exists(invoke_path):
        raise FileNotFoundError(f"invoke csv not found: {invoke_path}")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"train csv not found: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"test csv not found: {test_path}")

    invoke_df = pd.read_csv(invoke_path)
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    g, all_nodes, num_mashup = build_invoke_graph_from_df(invoke_df)
    train_g, train_pos_g, train_neg_g, test_pos_g, test_neg_g = build_subgraphs_from_train_test(
        train_df,
        test_df,
        num_mashup=num_mashup,
        all_nodes=all_nodes,
        neg_sample_size=neg_sample_size,
        invoke_df=invoke_df,
    )

    return train_g, train_pos_g, train_neg_g, test_pos_g, test_neg_g, all_nodes, num_mashup, g


# ================================================================
# 2. MULTI-LEVEL HYPERGRAPHS
# ================================================================

def build_co_service_and_co_mashup_hypergraph(
    g: dgl.DGLGraph,
    num_mashup: int,
    num_api: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build two dense hypergraph incidence matrices from the base bipartite graph:

    - Co-service hypergraph H_cs (mashup × hyperedge):
        each API induces a hyperedge over mashups that invoke it.

    - Co-mashup hypergraph H_cm (api × hyperedge):
        each mashup induces a hyperedge over APIs it invokes.
    """
    src, dst = g.edges()
    src = src.numpy()
    dst = dst.numpy() - num_mashup  # map API nodes to [0, num_api)

    M = torch.zeros((num_mashup, num_api), dtype=torch.float32)
    M[src, dst] = 1.0

    # Co-service: hyperedges over mashups
    rows_cs: List[int] = []
    cols_cs: List[int] = []
    edge_id = 0
    for j in range(num_api):
        mashup_ids = (M[:, j] == 1).nonzero(as_tuple=True)[0]
        if len(mashup_ids) >= 2:
            for i in mashup_ids:
                rows_cs.append(i.item())
                cols_cs.append(edge_id)
            edge_id += 1

    H_cs = torch.sparse_coo_tensor(
        indices=[rows_cs, cols_cs],
        values=torch.ones(len(rows_cs)),
        size=(num_mashup, edge_id),
    ).to_dense()

    # Co-mashup: hyperedges over APIs
    rows_cm: List[int] = []
    cols_cm: List[int] = []
    edge_id = 0
    for i in range(num_mashup):
        api_ids = (M[i, :] == 1).nonzero(as_tuple=True)[0]
        if len(api_ids) >= 2:
            for j in api_ids:
                rows_cm.append(j.item())
                cols_cm.append(edge_id)
            edge_id += 1

    H_cm = torch.sparse_coo_tensor(
        indices=[rows_cm, cols_cm],
        values=torch.ones(len(rows_cm)),
        size=(num_api, edge_id),
    ).to_dense()

    return H_cs, H_cm


def build_co_tag_hypergraph_dense(df: pd.DataFrame, column_name: str) -> torch.Tensor:
    """
    Build a dense co-tag hypergraph incidence matrix.

    Each distinct tag induces a hyperedge. The matrix shape is:
        H_tag ∈ R^{num_nodes × num_tags}

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing a tag column stored as a stringified Python list.
    column_name : str
        Name of the tag column (e.g., 'MashupCategory' or 'ApiTags').

    Returns
    -------
    torch.Tensor
        Dense hypergraph incidence matrix.
    """
    num_nodes = len(df)
    tag_to_nodes: Dict[str, List[int]] = {}

    for idx, row in df.iterrows():
        try:
            tags = ast.literal_eval(row[column_name])
        except Exception:
            continue
        for tag in tags:
            tag_to_nodes.setdefault(tag, []).append(idx)

    rows: List[int] = []
    cols: List[int] = []
    for edge_id, node_list in enumerate(tag_to_nodes.values()):
        for node in node_list:
            rows.append(node)
            cols.append(edge_id)

    H_tag = torch.sparse_coo_tensor(
        indices=[rows, cols],
        values=torch.ones(len(rows)),
        size=(num_nodes, len(tag_to_nodes)),
    ).to_dense()

    return H_tag


def build_co_provider_hypergraph_dense(
    df: pd.DataFrame,
    column_name: str = "ApiProvider",
) -> torch.Tensor:
    """
    Build a dense co-provider hypergraph incidence matrix.

    Each distinct provider induces a hyperedge. The matrix shape is:
        H_provider ∈ R^{num_nodes × num_providers}
    """
    num_nodes = len(df)
    provider_to_nodes: Dict[str, List[int]] = {}

    for idx, row in df.iterrows():
        provider = str(row[column_name]).strip()
        if provider:
            provider_to_nodes.setdefault(provider, []).append(idx)

    rows: List[int] = []
    cols: List[int] = []
    for edge_id, node_list in enumerate(provider_to_nodes.values()):
        for node in node_list:
            rows.append(node)
            cols.append(edge_id)

    H_provider = torch.sparse_coo_tensor(
        indices=[rows, cols],
        values=torch.ones(len(rows)),
        size=(num_nodes, len(provider_to_nodes)),
    ).to_dense()

    return H_provider


def build_co_description_hypergraph_by_id(
    json_path: str,
    df: pd.DataFrame,
    top_k: int = 3,
    entity_type: str = "mashup",
) -> torch.Tensor:
    """
    Construct a description-based hypergraph using BERT embeddings.

    For each entity (mashup or API), we build one hyperedge connecting the
    entity itself and its top-k most similar neighbors under cosine similarity.

    Parameters
    ----------
    json_path : str
        Path to a JSON file mapping names to BERT embeddings: {name: [float, ...]}.
    df : pd.DataFrame
        DataFrame containing ordered IDs and names.
    top_k : int, optional
        Number of most similar neighbors to connect in each hyperedge.
    entity_type : {'mashup', 'api'}
        Whether the DataFrame represents mashups or APIs.

    Returns
    -------
    torch.Tensor
        Dense incidence matrix H_desc.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        name_to_emb = json.load(f)

    df = df.sort_values("ID")
    if entity_type == "mashup":
        names_in_order = df["MashupName"].tolist()
    elif entity_type == "api":
        names_in_order = df["ApiName"].tolist()
    else:
        raise ValueError("entity_type must be either 'mashup' or 'api'.")

    embeddings: List[torch.Tensor] = []
    for name in names_in_order:
        if name not in name_to_emb:
            raise ValueError(f"BERT embedding not found for: {name}")
        emb = torch.tensor(name_to_emb[name], dtype=torch.float32)
        embeddings.append(emb)

    embeddings_t = torch.stack(embeddings, dim=0)
    num_nodes = embeddings_t.size(0)

    norm_emb = F.normalize(embeddings_t, dim=1)
    sim_matrix = torch.matmul(norm_emb, norm_emb.T)

    rows: List[int] = []
    cols: List[int] = []
    edge_id = 0
    for i in range(num_nodes):
        sim_matrix[i, i] = -1.0
        topk = torch.topk(sim_matrix[i], k=top_k).indices.tolist()
        hyperedge_nodes = [i] + topk
        for node in hyperedge_nodes:
            rows.append(node)
            cols.append(edge_id)
        edge_id += 1

    H_desc = torch.sparse_coo_tensor(
        indices=[rows, cols],
        values=torch.ones(len(rows)),
        size=(num_nodes, edge_id),
    ).to_dense()

    return H_desc


def build_api_bundle_hypergraph_dense(
    bundle_csv_path: str,
    num_api: int,
    min_bundle_size: int = 1,
) -> torch.Tensor:
    """
    Build a dense API hypergraph incidence matrix from bundle mining results.

    CSV format
    ----------
    The file must contain a column named 'itemsets', where each row is a
    comma-separated list of API IDs, e.g.
        itemsets
        "2,12"
        "0,2"
        "0,47"
        ...

    Parameters
    ----------
    bundle_csv_path : str
        Path to the bundle csv file (no directory nesting assumed).
    num_api : int
        Total number of APIs.
    min_bundle_size : int, optional
        Minimum bundle cardinality to be kept.

    Returns
    -------
    torch.Tensor
        Dense incidence matrix H_api ∈ R^{num_api × num_bundles}.
    """
    if not os.path.exists(bundle_csv_path):
        raise FileNotFoundError(f"bundle file not found: {bundle_csv_path}")

    df = pd.read_csv(bundle_csv_path)
    if "itemsets" not in df.columns:
        raise KeyError("CSV must contain a column named 'itemsets'.")

    hyperedges: List[List[int]] = []
    for raw_val in df["itemsets"]:
        if pd.isna(raw_val):
            continue

        cell = str(raw_val).strip()
        if (cell.startswith('"') and cell.endswith('"')) or (
            cell.startswith("'") and cell.endswith("'")
        ):
            cell = cell[1:-1].strip()
        if not cell:
            continue

        tokens = [t.strip() for t in cell.split(",") if t.strip()]
        if not all(tok.lstrip("-").isdigit() for tok in tokens):
            continue

        api_ids = sorted(set(int(tok) for tok in tokens))
        if len(api_ids) < min_bundle_size:
            continue

        api_ids = [aid for aid in api_ids if 0 <= aid < num_api]
        if not api_ids:
            continue

        hyperedges.append(api_ids)

    num_bundles = len(hyperedges)
    if num_bundles == 0:
        raise ValueError("No valid bundles found in CSV. Check data or min_bundle_size.")

    H = torch.zeros((num_api, num_bundles), dtype=torch.float32)
    for b_idx, api_ids in enumerate(hyperedges):
        for aid in api_ids:
            H[aid, b_idx] = 1.0

    return H


def build_mashup_bundle_hypergraph_dense(
    invoke_df: pd.DataFrame,
    bundle_csv_path: str,
    num_mashup: int,
    num_api: int,
    theta: float = 0.6,
    min_bundle_size: int = 1,
) -> torch.Tensor:
    """
    Build a weighted mashup–bundle hypergraph H_m.

    For each mashup i and each bundle b_k, the weight is defined as:
        cover(i, k) = |S_i ∩ b_k| / |b_k|
    where S_i is the set of APIs used by mashup i.

    We only keep cover(i, k) >= theta.

    Parameters
    ----------
    invoke_df : pd.DataFrame
        Must contain columns ['Mashup_ID', 'Api_ID'].
    bundle_csv_path : str
        Path to the bundle csv file (same format as for API hypergraph).
    num_mashup : int
        Number of mashup nodes.
    num_api : int
        Number of API nodes.
    theta : float, optional
        Coverage threshold.
    min_bundle_size : int, optional
        Minimum bundle cardinality.

    Returns
    -------
    torch.Tensor
        H_m ∈ R^{num_mashup × num_bundles} with coverage weights.
    """
    if not os.path.exists(bundle_csv_path):
        raise FileNotFoundError(f"bundle file not found: {bundle_csv_path}")

    if "Mashup_ID" not in invoke_df.columns or "Api_ID" not in invoke_df.columns:
        raise KeyError("invoke_df must contain 'Mashup_ID' and 'Api_ID' columns.")

    mashup_apis: List[set[int]] = [set() for _ in range(num_mashup)]
    for _, row in invoke_df.iterrows():
        m_id = int(row["Mashup_ID"])
        a_id = int(row["Api_ID"])
        if 0 <= m_id < num_mashup and 0 <= a_id < num_api:
            mashup_apis[m_id].add(a_id)

    df_bundle = pd.read_csv(bundle_csv_path)
    if "itemsets" not in df_bundle.columns:
        raise KeyError("CSV must contain a column named 'itemsets'.")

    bundles: List[set[int]] = []
    for raw_val in df_bundle["itemsets"]:
        if pd.isna(raw_val):
            continue
        cell = str(raw_val).strip()
        if (cell.startswith('"') and cell.endswith('"')) or (
            cell.startswith("'") and cell.endswith("'")
        ):
            cell = cell[1:-1].strip()
        if not cell:
            continue

        tokens = [t.strip() for t in cell.split(",") if t.strip()]
        if not all(tok.lstrip("-").isdigit() for tok in tokens):
            continue

        api_ids = sorted(set(int(tok) for tok in tokens))
        if len(api_ids) < min_bundle_size:
            continue

        api_ids = [aid for aid in api_ids if 0 <= aid < num_api]
        if not api_ids:
            continue

        bundles.append(set(api_ids))

    num_bundles = len(bundles)
    if num_bundles == 0:
        raise ValueError("No valid bundles found in CSV. Check data or min_bundle_size.")

    H_m = torch.zeros((num_mashup, num_bundles), dtype=torch.float32)

    for m_id in range(num_mashup):
        S_i = mashup_apis[m_id]
        if not S_i:
            continue
        for b_idx, bundle in enumerate(bundles):
            inter_size = len(S_i & bundle)
            if inter_size == 0:
                continue
            cover = inter_size / len(bundle)
            if cover >= theta:
                H_m[m_id, b_idx] = cover

    return H_m


def build_multi_level_hyperedges(
    g: dgl.DGLGraph,
    data_path: str,
    num_mashup: int,
    num_api: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build the multi-level hypergraph incidence matrices H_m and H_a.

    H_m aggregates hyperedges defined over mashups (co-service, category,
    description-based), while H_a aggregates hyperedges defined over APIs
    (co-mashup, category, provider, description-based).
    """
    src, dst = g.edges()
    _ = src  # src/dst are not used directly here; g is passed to internal builders
    dst = dst - num_mashup  # API ids, if needed later

    mashup_path = os.path.join(data_path, "mashup_data.csv")
    api_path = os.path.join(data_path, "api_data.csv")

    if not os.path.exists(mashup_path):
        raise FileNotFoundError(f"mashup_data.csv not found in {data_path}")
    if not os.path.exists(api_path):
        raise FileNotFoundError(f"api_data.csv not found in {data_path}")

    mashup_df = pd.read_csv(mashup_path)
    api_df = pd.read_csv(api_path)

    H_cs, H_cm = build_co_service_and_co_mashup_hypergraph(g, num_mashup, num_api)

    H_m_cc = build_co_tag_hypergraph_dense(mashup_df, "MashupCategory")
    H_a_cc = build_co_tag_hypergraph_dense(api_df, "ApiTags")

    H_a_cp = build_co_provider_hypergraph_dense(api_df, "ApiProvider")

    mashup_bert_json = os.path.join(data_path, "bert_mashup_des.json")
    api_bert_json = os.path.join(data_path, "bert_api_des.json")
    H_m_desc = build_co_description_hypergraph_by_id(
        mashup_bert_json, mashup_df, top_k=3, entity_type="mashup"
    )
    H_a_desc = build_co_description_hypergraph_by_id(
        api_bert_json, api_df, top_k=3, entity_type="api"
    )

    H_m = torch.cat([H_cs, H_m_cc, H_m_desc], dim=1)
    H_a = torch.cat([H_cm, H_a_cc, H_a_cp, H_a_desc], dim=1)

    return H_m, H_a


# ================================================================
# 3. SIMPLE FEATURE ENCODING (OPTIONAL)
# ================================================================

def encode_node_features(data_path: str):
    """
    Encode basic categorical features for mashups and APIs using one-hot
    and multi-hot encoders, plus pre-computed BERT description embeddings.

    This is optional and can be used to initialize node features before
    graph/ hypergraph message passing.
    """
    feat_types = ["oneHot", "multiHot"]
    feat_config = {
        "mashup": {
            "oneHot": ["ID", "MashupName"],
            "multiHot": ["MashupCategory"],
            "textual": ["MashupDescription"],
        },
        "api": {
            "oneHot": ["ID", "ApiName"],
            "multiHot": ["ApiTags"],
            "textual": ["ApiDescription"],
        },
    }

    mashup_path = os.path.join(data_path, "mashup_data.csv")
    api_path = os.path.join(data_path, "api_data.csv")
    mashup_bert_path = os.path.join(data_path, "bert_mashup_des.json")
    api_bert_path = os.path.join(data_path, "bert_api_des.json")

    mashup_df = pd.read_csv(mashup_path)
    api_df = pd.read_csv(api_path)

    with open(mashup_bert_path, "r", encoding="utf-8") as f:
        mashup_bert = json.load(f)
    with open(api_bert_path, "r", encoding="utf-8") as f:
        api_bert = json.load(f)

    # ----- mashup side -----
    mashup_sumdata: Dict[str, List] = {}
    for feat_type in feat_types:
        for feat in feat_config["mashup"][feat_type]:
            mashup_sumdata[feat] = mashup_df[feat].tolist()

    mashup_encoders: Dict[str, MultiLabelBinarizer | LabelBinarizer] = {}
    for feat in feat_config["mashup"]["oneHot"]:
        enc = LabelBinarizer()
        enc.fit(mashup_sumdata[feat])
        mashup_encoders[feat] = enc
    for feat in feat_config["mashup"]["multiHot"]:
        enc = MultiLabelBinarizer()
        enc.fit(mashup_sumdata[feat])
        mashup_encoders[feat] = enc

    mashup_features: List[Dict[str, np.ndarray]] = [None] * mashup_df.shape[0]
    for idx, node in mashup_df.iterrows():
        node_feat: Dict[str, np.ndarray] = {}
        node_feat["MashupDescription"] = np.array(mashup_bert[node["MashupName"]])
        for feat_type in feat_types:
            for feat in feat_config["mashup"][feat_type]:
                enc = mashup_encoders[feat]
                encoded = enc.transform([node[feat]])
                node_feat[feat] = encoded
        mashup_features[idx] = node_feat

    # ----- api side -----
    api_sumdata: Dict[str, List] = {}
    for feat_type in feat_types:
        for feat in feat_config["api"][feat_type]:
            api_sumdata[feat] = api_df[feat].tolist()

    api_encoders: Dict[str, MultiLabelBinarizer | LabelBinarizer] = {}
    for feat in feat_config["api"]["oneHot"]:
        enc = LabelBinarizer()
        enc.fit(api_sumdata[feat])
        api_encoders[feat] = enc
    for feat in feat_config["api"]["multiHot"]:
        enc = MultiLabelBinarizer()
        enc.fit(api_sumdata[feat])
        api_encoders[feat] = enc

    api_features: List[Dict[str, np.ndarray]] = [None] * api_df.shape[0]
    for idx, node in api_df.iterrows():
        node_feat: Dict[str, np.ndarray] = {}
        node_feat["ApiDescription"] = np.array(api_bert[node["ApiName"]])
        for feat_type in feat_types:
            for feat in feat_config["api"][feat_type]:
                enc = api_encoders[feat]
                encoded = enc.transform([node[feat]])
                node_feat[feat] = encoded
        api_features[idx] = node_feat

    return mashup_features, api_features
