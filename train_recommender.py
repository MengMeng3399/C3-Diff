from parser import parse_args
import os
from typing import Dict, Tuple

import dgl
import torch
from dgl.data.utils import load_graphs
from build_guidance_signals import (
    build_bipartite_adjacency, build_mashup_guidance, build_api_guidance,
)

from recommender_model import (
    DotProductPredictor,
    Recommender,
    compute_auc,
    compute_f1,
    compute_loss,
    evaluation_metrics,
    set_seed,
)


def build_candidate_graph(test_pos_g: dgl.DGLGraph, num_mashup: int, num_api: int) -> dgl.DGLGraph:
    """
    Build a dense candidate graph for evaluation.

    The graph contains all candidate APIs for every mashup that appears in the test positive graph.
    """
    src_test, _ = test_pos_g.edges()
    unique_mashups = torch.unique(src_test)

    src = unique_mashups.repeat_interleave(num_api)
    api_ids = torch.arange(num_api, dtype=torch.long) + num_mashup
    dst = api_ids.repeat(len(unique_mashups))

    return dgl.graph((src, dst), num_nodes=num_mashup + num_api)


def load_data(data_dir: str) -> Tuple[Dict, Dict[str, torch.Tensor]]:
    """Load fixed, disjoint train/validation/test splits.

    graph.bin retains its existing layout: train_g, train_pos_g, train_neg_g,
    test_pos_g, test_neg_g, ... . validation.bin must contain exactly
    [val_pos_g, val_neg_g], using the same global node IDs. All splits must
    come from the same partition; a legacy train/test split must be rebuilt
    before adding validation. No test edges are repurposed as validation.
    """
    graph_path = os.path.join(data_dir, "graph.bin")
    glist, _ = load_graphs(graph_path)

    if len(glist) < 5:
        raise ValueError("graph.bin must contain training and test graphs")
    _, train_pos_g, train_neg_g, test_pos_g, test_neg_g = glist[:5]
    validation_path = os.path.join(data_dir, "validation.bin")
    if not os.path.isfile(validation_path):
        raise FileNotFoundError("Independent validation.bin is required; do not use test graphs for selection")
    validation, _ = load_graphs(validation_path)
    if len(validation) != 2:
        raise ValueError("validation.bin must contain [val_pos_g, val_neg_g]")
    val_pos_g, val_neg_g = validation
    train_g = dgl.to_bidirected(train_pos_g)

    mashup_content = torch.load(os.path.join(data_dir, "mashup_em.pt"), map_location="cpu")
    api_content = torch.load(os.path.join(data_dir, "api_em.pt"), map_location="cpu")
    num_mashup = len(mashup_content)
    num_api = len(api_content)
    validate_splits(train_pos_g, val_pos_g, test_pos_g, val_neg_g, test_neg_g,
                    num_mashup, num_api)
    # Derive guidance from exactly the same training edges as propagation.
    u, v = train_pos_g.edges()
    pairs = [(m, a - num_mashup) for m, a in zip(u.tolist(), v.tolist())]
    ma, am = build_bipartite_adjacency(pairs, num_mashup, num_api)
    mashup_guidance = build_mashup_guidance(ma, am, num_mashup)
    api_guidance = build_api_guidance(ma, am, num_mashup, num_api)

    n_params = {
        "n_mashup": num_mashup,
        "n_api": num_api,
        "mashup_content": mashup_content,
        "api_content": api_content,
        "guidance": {
            "mashup_guidance": mashup_guidance,
            "api_guidance": api_guidance,
        },
    }

    graphs = {
        "train_g": train_g,
        "train_pos_g": train_pos_g,
        "train_neg_g": train_neg_g,
        "val_pos_g": val_pos_g,
        "val_neg_g": val_neg_g,
        "test_pos_g": test_pos_g,
        "test_neg_g": test_neg_g,
    }
    return graphs, n_params


def move_graph_to_device(g: dgl.DGLGraph, device: torch.device) -> dgl.DGLGraph:
    """Move a DGL graph to the target device."""
    return g.to(device)


def sample_bpr_negatives(positive_g, train_g, num_mashup: int, num_api: int):
    """Sample an unobserved API for each positive edge, preserving its Mashup and order."""
    positive_src, positive_dst = positive_g.edges()
    observed = [set() for _ in range(num_mashup)]
    src, dst = train_g.edges()
    for u, v in zip(src.tolist(), dst.tolist()):
        if 0 <= u < num_mashup <= v < num_mashup + num_api:
            observed[u].add(v - num_mashup)
        elif 0 <= v < num_mashup <= u < num_mashup + num_api:
            observed[v].add(u - num_mashup)
    sources = positive_src.tolist()
    for u, v in zip(sources, positive_dst.tolist()):
        if not (0 <= u < num_mashup <= v < num_mashup + num_api):
            raise ValueError("Positive edges must be Mashup -> API")
        observed[u].add(v - num_mashup)
    negatives = torch.empty_like(positive_dst)
    for u in sorted(set(sources)):
        available = sorted(set(range(num_api)) - observed[u])
        if not available:
            raise ValueError(f"Mashup {u} has no unobserved API for BPR")
        choices = torch.tensor(available, device=positive_src.device)
        positions = (positive_src == u).nonzero(as_tuple=True)[0]
        draws = torch.randint(len(available), (positions.numel(),), device=positive_src.device)
        negatives[positions] = choices[draws] + num_mashup
    return dgl.graph((positive_src, negatives), num_nodes=num_mashup + num_api,
                     device=positive_src.device)


def validate_splits(train, val, test, val_neg, test_neg, n_mashup, n_api):
    def edges(g):
        if g.num_nodes() != n_mashup + n_api:
            raise ValueError("All split graphs must use the full global node space")
        u, v = g.edges()
        pairs = list(zip(u.tolist(), v.tolist()))
        if not pairs or any(not (0 <= m < n_mashup <= a < n_mashup + n_api) for m, a in pairs):
            raise ValueError("Nonempty Mashup -> API split graphs are required")
        if len(set(pairs)) != len(pairs):
            raise ValueError("Split graphs must not contain duplicate edges")
        return set(pairs)
    train_edges, val_edges, test_edges = map(edges, (train, val, test))
    if train_edges & val_edges or train_edges & test_edges or val_edges & test_edges:
        raise ValueError("Training, validation and test positives must be disjoint")
    observed = train_edges | val_edges | test_edges
    if edges(val_neg) & observed or edges(test_neg) & observed:
        raise ValueError("Evaluation negatives must not be observed positive interactions")


def iter_positive_batches(graph, batch_size):
    """Shuffle positive interactions; retain the final partial batch."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    src, dst = graph.edges()
    order = torch.randperm(src.numel(), device=src.device)
    for start in range(0, src.numel(), batch_size):
        ids = order[start:start + batch_size]
        yield dgl.graph((src[ids], dst[ids]), num_nodes=graph.num_nodes(), device=src.device)


@torch.no_grad()
def evaluate(model, predictor, train_g, train_pos_g, pos_g, neg_g, n_mashup, n_api, top_k, seed):
    model.eval()
    predictor.eval()
    # Fix the evaluation noise without perturbing the training RNG stream.
    devices = [next(model.parameters()).device.index] if next(model.parameters()).is_cuda else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        h, _ = model(train_g)
    candidates = build_candidate_graph(pos_g.to("cpu"), n_mashup, n_api).to(h.device)
    return evaluation_metrics(candidates, pos_g, predictor(candidates, h),
        {"n_mashup": n_mashup, "n_api": n_api}, predictor(pos_g, h),
        predictor(neg_g, h), top_k, train_pos_g=train_pos_g)


def main() -> None:
    args = parse_args()

    set_seed(args.seed)

    if args.cuda and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu_id}")
    else:
        device = torch.device("cpu")

    graphs, n_params = load_data(args.data_dir)
    num_mashup = n_params["n_mashup"]
    num_api = n_params["n_api"]

    n_params.update(
        {
            "hidden_dim": args.hidden_dim,
            "hops": tuple(range(1, args.max_hops + 1)),
            "num_layers": args.num_layers,
            "diffusion_steps": args.diffusion_steps,
            "dropout": args.dropout,
            "lambda_diff": args.lambda_diff,
            "lambda_cl": args.lambda_cl,
            "temperature": args.temperature,
            "beta_start": args.beta_start,
            "beta_end": args.beta_end,
        }
    )

    model = Recommender(n_params).to(device)
    predictor = DotProductPredictor().to(device)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(predictor.parameters()),
        lr=args.lr,
    )

    train_g = move_graph_to_device(graphs["train_g"], device)
    train_pos_g = move_graph_to_device(graphs["train_pos_g"], device)
    val_pos_g = move_graph_to_device(graphs["val_pos_g"], device)
    val_neg_g = move_graph_to_device(graphs["val_neg_g"], device)
    best_recall = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        predictor.train()
        epoch_loss, steps = 0.0, 0
        for batch_pos_g in iter_positive_batches(train_pos_g, args.batch_size):
            # Refresh the full graph representation at each optimizer step;
            # recommendation triplets are minibatched, not graph neighborhoods.
            h, loss_reg = model(train_g)
            negative_g = sample_bpr_negatives(batch_pos_g, train_g, num_mashup, num_api)
            loss, _ = compute_loss(predictor(batch_pos_g, h), predictor(negative_g, h),
                                   loss_reg, model.parameters(), args.l2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            steps += 1

        if epoch % args.eval_every == 0 or epoch == 1 or epoch == args.epochs:
            metrics = evaluate(model, predictor, train_g, train_pos_g, val_pos_g, val_neg_g,
                               num_mashup, num_api, args.top_k, args.seed)
            recall, ndcg, f1, auc = metrics
            print(f"Epoch {epoch:03d} | train_loss={epoch_loss / max(steps, 1):.4f} | "
                  f"validation Recall@{args.top_k}={recall:.4f} | NDCG={ndcg:.4f} | "
                  f"F1={f1:.4f} | AUC={auc:.4f}")
            if recall > best_recall:
                best_recall = recall
                torch.save({"model_state_dict": model.state_dict(), "args": vars(args),
                            "selection_split": "validation", "epoch": epoch,
                            "validation_metrics": metrics}, args.save_path)

    # Restore the validation-selected checkpoint before the sole test evaluation.
    checkpoint = torch.load(args.save_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate(model, predictor, train_g, train_pos_g,
                            graphs["test_pos_g"].to(device), graphs["test_neg_g"].to(device),
                            num_mashup, num_api, args.top_k, args.seed)
    print(f"Selected epoch {checkpoint['epoch']} using validation Recall; "
          f"test Recall@{args.top_k}={test_metrics[0]:.4f} | "
          f"NDCG={test_metrics[1]:.4f} | F1={test_metrics[2]:.4f} | AUC={test_metrics[3]:.4f}")


if __name__ == "__main__":
    main()




