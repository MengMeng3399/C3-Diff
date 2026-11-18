import argparse
import os


# Table 2 and the connectivity-order analysis.
DATASET_DEFAULTS = {
    "PWA": {"lr": 1e-3, "l2": 1e-3, "lambda_diff": 0.1, "max_hops": 2},
    "HGA": {"lr": 3e-4, "l2": 1e-5, "lambda_diff": 0.2, "max_hops": 3},
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="C3-Diff")
    parser.add_argument("--dataset", choices=("PWA", "HGA"), default="PWA")
    parser.add_argument("--data_dir", type=str, default=None, help="Directory of processed graph/features.")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU id to use when CUDA is available.")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA if available.")
    parser.add_argument("--seed", type=int, default=2020, help="Random seed.")
    parser.add_argument("--batch_size", type=int, default=128, help="Positive interactions per optimizer step.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--eval_every", type=int, default=5, help="Evaluation interval.")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension.")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of LightGCN layers.")
    parser.add_argument("--diffusion_steps", type=int, default=5, help="Number of diffusion steps.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate.")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate.")
    parser.add_argument("--l2", type=float, default=None, help="L2 regularization weight.")
    parser.add_argument("--lambda_diff", type=float, default=None, help="Weight of diffusion reconstruction loss.")
    parser.add_argument("--lambda_cl", type=float, default=0.05, help="Weight of alignment loss.")
    parser.add_argument("--temperature", type=float, default=0.2, help="Temperature for InfoNCE.")
    parser.add_argument("--beta_start", type=float, default=1e-4, help="Diffusion beta start.")
    parser.add_argument("--beta_end", type=float, default=2e-2, help="Diffusion beta end.")
    parser.add_argument("--top_k", type=int, default=10, help="Top-k for ranking metrics.")
    parser.add_argument("--save_path", type=str, default="./best_recommender.pt", help="Path to save the best model checkpoint.")
    parser.add_argument("--max_hops", type=int, choices=(1, 2, 3), default=None)
    args = parser.parse_args(argv)
    for name, value in DATASET_DEFAULTS[args.dataset].items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    if args.data_dir is None:
        args.data_dir = os.path.join("data", args.dataset)
    args.data_path = args.data_dir
    if min(args.epochs, args.batch_size, args.eval_every, args.top_k) <= 0:
        parser.error("epochs, batch_size, eval_every and top_k must be positive")

    if min(args.hidden_dim, args.diffusion_steps, args.lr, args.temperature) <= 0 or args.num_layers < 0:
        parser.error("Dimensions, steps, learning rate and temperature must be positive")
    if min(args.l2, args.lambda_diff, args.lambda_cl) < 0 or not 0 <= args.dropout < 1:
        parser.error("Loss weights must be nonnegative and dropout must be in [0, 1)")
    if not 0 < args.beta_start <= args.beta_end < 1:
        parser.error("Require 0 < beta_start <= beta_end < 1")
    return args
