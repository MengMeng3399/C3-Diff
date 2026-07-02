import argparse
import os
from typing import Dict, Tuple

import dgl
import torch
from dgl.data.utils import load_graphs

from recommender_model import (
    MLPPredictor,
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
    """Load graphs, content features, and guidance tensors from disk."""
    graph_path = os.path.join(data_dir, "graph.bin")
    glist, _ = load_graphs(graph_path)

    train_g, train_pos_g, train_neg_g, test_pos_g, test_neg_g, _ = glist[:6]

    mashup_content = torch.load(os.path.join(data_dir, "mashup_em.pt"), map_location="cpu")
    api_content = torch.load(os.path.join(data_dir, "api_em.pt"), map_location="cpu")
    mashup_guidance = torch.load(os.path.join(data_dir, "mashup_guidance_train.pt"), map_location="cpu")
    api_guidance = torch.load(os.path.join(data_dir, "api_guidance_train.pt"), map_location="cpu")

    num_mashup = len(mashup_content)
    num_api = len(api_content)

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
        "test_pos_g": test_pos_g,
        "test_neg_g": test_neg_g,
    }
    return graphs, n_params


def move_graph_to_device(g: dgl.DGLGraph, device: torch.device) -> dgl.DGLGraph:
    """Move a DGL graph to the target device."""
    return g.to(device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the mashup-API recommender.")
    parser.add_argument("--data_dir", type=str, default="./data/PWA", help="Directory of processed graph/features.")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU id to use when CUDA is available.")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA if available.")
    parser.add_argument("--seed", type=int, default=2020, help="Random seed.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--eval_every", type=int, default=5, help="Evaluation interval.")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension.")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of LightGCN layers.")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads in denoiser.")
    parser.add_argument("--diffusion_steps", type=int, default=10, help="Number of diffusion steps.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--l2", type=float, default=1e-4, help="L2 regularization weight.")
    parser.add_argument("--lambda_diff", type=float, default=1.0, help="Weight of diffusion reconstruction loss.")
    parser.add_argument("--lambda_cl", type=float, default=0.1, help="Weight of alignment loss.")
    parser.add_argument("--temperature", type=float, default=0.2, help="Temperature for InfoNCE.")
    parser.add_argument("--beta_start", type=float, default=1e-4, help="Diffusion beta start.")
    parser.add_argument("--beta_end", type=float, default=2e-2, help="Diffusion beta end.")
    parser.add_argument("--top_k", type=int, default=10, help="Top-k for ranking metrics.")
    parser.add_argument("--save_path", type=str, default="./best_recommender.pt", help="Path to save the best model checkpoint.")
    args = parser.parse_args()

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
            "num_layers": args.num_layers,
            "num_heads": args.num_heads,
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
    predictor = MLPPredictor(args.hidden_dim).to(device)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(predictor.parameters()),
        lr=args.lr,
    )

    train_g = move_graph_to_device(graphs["train_g"], device)
    train_pos_g = move_graph_to_device(graphs["train_pos_g"], device)
    train_neg_g = move_graph_to_device(graphs["train_neg_g"], device)
    test_pos_g = move_graph_to_device(graphs["test_pos_g"], device)
    test_neg_g = move_graph_to_device(graphs["test_neg_g"], device)

    candidate_g = build_candidate_graph(graphs["test_pos_g"], num_mashup, num_api).to(device)

    best_recall = -1.0
    best_metrics = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        predictor.train()

        h, loss_reg = model(train_g)
        pos_score = predictor(train_pos_g, h)
        neg_score = predictor(train_neg_g, h)

        loss, ranking_loss_only = compute_loss(
            pos_score=pos_score,
            neg_score=neg_score,
            loss_reg=loss_reg,
            h=h,
            n_mashup=num_mashup,
            l2_weight=args.l2,
            reg_weight=0.1,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % args.eval_every == 0 or epoch == 1 or epoch == args.epochs:
            model.eval()
            predictor.eval()

            with torch.no_grad():
                h_eval, _ = model(train_g)
                candidate_score = predictor(candidate_g, h_eval)
                test_pos_score = predictor(test_pos_g, h_eval)
                test_neg_score = predictor(test_neg_g, h_eval)

                recall, ndcg, f1, auc = evaluation_metrics(
                    candidate_g=candidate_g,
                    pos_g=test_pos_g,
                    score=candidate_score,
                    data_config={"n_mashup": num_mashup, "n_api": num_api},
                    pos_score=test_pos_score,
                    neg_score=test_neg_score,
                    top_k=args.top_k,
                )

            print(
                f"Epoch {epoch:03d} | "
                f"train_loss={loss.item():.4f} | "
                f"rank_loss={ranking_loss_only.item():.4f} | "
                f"Recall@{args.top_k}={recall:.4f} | "
                f"NDCG@{args.top_k}={ndcg:.4f} | "
                f"F1={f1:.4f} | "
                f"AUC={auc:.4f}"
            )

            if recall > best_recall:
                best_recall = recall
                best_metrics = {
                    "epoch": epoch,
                    "recall": recall,
                    "ndcg": ndcg,
                    "f1": f1,
                    "auc": auc,
                }
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "predictor_state_dict": predictor.state_dict(),
                        "args": vars(args),
                        "best_metrics": best_metrics,
                    },
                    args.save_path,
                )

    if best_metrics is not None:
        print("\nBest checkpoint summary:")
        print(
            f"Epoch {best_metrics['epoch']} | "
            f"Recall@{args.top_k}={best_metrics['recall']:.4f} | "
            f"NDCG@{args.top_k}={best_metrics['ndcg']:.4f} | "
            f"F1={best_metrics['f1']:.4f} | "
            f"AUC={best_metrics['auc']:.4f}"
        )
        print(f"Checkpoint saved to: {args.save_path}")


if __name__ == "__main__":
    main()
