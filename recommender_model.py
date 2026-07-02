import math
from typing import Dict, List, Tuple, Union

import dgl
import dgl.function as fn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, ndcg_score, roc_auc_score


TensorLike = Union[str, torch.Tensor]


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_auc(pos_score: torch.Tensor, neg_score: torch.Tensor) -> float:
    """Compute AUC from positive and negative edge scores."""
    scores = torch.cat([pos_score, neg_score]).detach().cpu().numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]
    ).detach().cpu().numpy()
    return float(roc_auc_score(labels, scores))


def compute_f1(pos_score: torch.Tensor, neg_score: torch.Tensor, threshold: float = 0.0) -> float:
    """Compute binary F1 after thresholding logits."""
    scores = torch.cat([pos_score, neg_score]).detach().cpu().numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]
    ).detach().cpu().numpy()
    preds = (scores >= threshold).astype(np.int64)
    return float(f1_score(labels, preds))


def top_k_list(
    g: dgl.DGLGraph,
    score: torch.Tensor,
    n_mashup: int,
    n_api: int,
    top_k: int = 10,
) -> np.ndarray:
    """
    Build a dense score matrix and return top-k API indices for each mashup.

    The graph is assumed to contain mashup -> API candidate edges.
    """
    top_k = min(top_k, n_api)
    score_matrix = np.zeros((n_mashup, n_api), dtype=np.float32)

    src, dst = g.edges()
    src = src.detach().cpu().numpy()
    dst = dst.detach().cpu().numpy()
    score_np = score.detach().cpu().numpy()

    for i in range(len(src)):
        mashup_id = int(src[i])
        api_id = int(dst[i] - n_mashup)
        if 0 <= mashup_id < n_mashup and 0 <= api_id < n_api:
            score_matrix[mashup_id, api_id] = score_np[i]

    top_k_index = np.argpartition(score_matrix, -top_k, axis=1)[:, -top_k:]
    return top_k_index


def recall_at_k_user_level(
    candidate_g: dgl.DGLGraph,
    pos_g: dgl.DGLGraph,
    score: torch.Tensor,
    data_config: Dict[str, int],
    top_k: int,
) -> float:
    """
    Compute user-level Recall@K.

    For each mashup, recall is:
        retrieved positive APIs in top-k / all positive APIs
    The final score is the average over mashups that have at least one positive API.
    """
    n_mashup = data_config["n_mashup"]
    n_api = data_config["n_api"]

    top_k_index = top_k_list(candidate_g, score, n_mashup, n_api, top_k)
    topk_sets = [set(row.tolist()) for row in top_k_index]

    src, dst = pos_g.edges()
    mashup_apis: List[List[int]] = [[] for _ in range(n_mashup)]
    for s, d in zip(src.tolist(), dst.tolist()):
        api_id = int(d) - n_mashup
        if 0 <= api_id < n_api:
            mashup_apis[int(s)].append(api_id)

    recalls = []
    for mashup_id in range(n_mashup):
        true_apis = set(mashup_apis[mashup_id])
        if not true_apis:
            continue
        hit_num = len(topk_sets[mashup_id] & true_apis)
        recalls.append(hit_num / len(true_apis))

    return float(sum(recalls) / len(recalls)) if recalls else 0.0


def compute_recall(
    candidate_g: dgl.DGLGraph,
    pos_g: dgl.DGLGraph,
    score: torch.Tensor,
    data_config: Dict[str, int],
    top_k: int,
) -> float:
    """
    Compute pair-level hit ratio at K.

    For each positive pair (mashup, api), count whether the api appears
    in the top-k predictions of the mashup, then average over all pairs.
    """
    n_mashup = data_config["n_mashup"]
    n_api = data_config["n_api"]

    top_k_index = top_k_list(candidate_g, score, n_mashup, n_api, top_k)
    topk_sets = [set(row.tolist()) for row in top_k_index]

    src, dst = pos_g.edges()
    hit = 0
    total = 0

    for s, d in zip(src.tolist(), dst.tolist()):
        mashup_id = int(s)
        api_id = int(d) - n_mashup
        if 0 <= api_id < n_api:
            total += 1
            if api_id in topk_sets[mashup_id]:
                hit += 1

    return float(hit / total) if total > 0 else 0.0


def ndcg_at_k(
    candidate_g: dgl.DGLGraph,
    pos_g: dgl.DGLGraph,
    score: torch.Tensor,
    data_config: Dict[str, int],
    top_k: int,
) -> float:
    """
    Compute NDCG@K for mashups that appear in the positive test graph.
    """
    src_pos, dst_pos = pos_g.edges()
    unique_mashups = torch.unique(src_pos).tolist()
    mapped_u = {value: index for index, value in enumerate(unique_mashups)}

    n_eval_mashup = len(unique_mashups)
    n_mashup = data_config["n_mashup"]
    n_api = data_config["n_api"]

    label_matrix = np.zeros((n_eval_mashup, n_api), dtype=np.float32)
    for s, d in zip(src_pos.tolist(), dst_pos.tolist()):
        label_matrix[mapped_u[int(s)], int(d) - n_mashup] = 1.0

    src_cand, dst_cand = candidate_g.edges()
    score_matrix = np.zeros((n_eval_mashup, n_api), dtype=np.float32)
    score_np = score.detach().cpu().numpy()
    for i, (s, d) in enumerate(zip(src_cand.tolist(), dst_cand.tolist())):
        if int(s) not in mapped_u:
            continue
        api_id = int(d) - n_mashup
        if 0 <= api_id < n_api:
            score_matrix[mapped_u[int(s)], api_id] = max(float(score_np[i]), 0.0)

    return float(ndcg_score(label_matrix, score_matrix, k=min(top_k, n_api)))


def evaluation_metrics(
    candidate_g: dgl.DGLGraph,
    pos_g: dgl.DGLGraph,
    score: torch.Tensor,
    data_config: Dict[str, int],
    pos_score: torch.Tensor,
    neg_score: torch.Tensor,
    top_k: int,
) -> Tuple[float, float, float, float]:
    """Return Recall@K, NDCG@K, F1, and AUC."""
    recall = recall_at_k_user_level(candidate_g, pos_g, score, data_config, top_k)
    ndcg = ndcg_at_k(candidate_g, pos_g, score, data_config, top_k)
    f1 = compute_f1(pos_score, neg_score)
    auc = compute_auc(pos_score, neg_score)
    return recall, ndcg, f1, auc


def _load_tensor(x: TensorLike, name: str) -> torch.Tensor:
    """
    Load a feature tensor from memory or a .pt file.

    Supported inputs:
        1) torch.Tensor
        2) path to a .pt file containing a tensor
    """
    if torch.is_tensor(x):
        return x.float()

    if isinstance(x, str):
        obj = torch.load(x, map_location="cpu")
        if torch.is_tensor(obj):
            return obj.float()
        raise ValueError(f"{name} file must contain a tensor, but got {type(obj)}")

    raise ValueError(f"{name} must be a tensor or a file path, but got {type(x)}")


def _build_padded_guidance(guidance_list: List[dict], hop_key: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert variable-length guidance neighbors into padded tensors.

    Returns:
        idx:     [N, L] global node ids, padded with -1
        weights: [N, L] normalized weights
        mask:    [N, L] valid positions
    """
    num_nodes = len(guidance_list)
    max_len = 0
    for item in guidance_list:
        hop_info = item.get(hop_key, {})
        global_ids = hop_info.get("global_ids", torch.empty(0, dtype=torch.long))
        max_len = max(max_len, int(global_ids.numel()))

    idx = torch.full((num_nodes, max_len), -1, dtype=torch.long)
    weights = torch.zeros((num_nodes, max_len), dtype=torch.float32)
    mask = torch.zeros((num_nodes, max_len), dtype=torch.bool)

    for i, item in enumerate(guidance_list):
        hop_info = item.get(hop_key, {})
        ids = hop_info.get("global_ids", torch.empty(0, dtype=torch.long)).long()
        w = hop_info.get("weights", torch.empty(0, dtype=torch.float32)).float()

        if ids.numel() == 0:
            continue
        if ids.numel() != w.numel():
            raise ValueError(f"{hop_key} global_ids and weights length mismatch at sample {i}")

        length = ids.numel()
        idx[i, :length] = ids
        weights[i, :length] = w
        mask[i, :length] = True

    return idx, weights, mask


def _build_dense_guidance(guidance_list: List[dict], hop_key: str, out_dim: int) -> torch.Tensor:
    """
    Build dense hop guidance features from local ids and weights.

    Returns:
        dense_tensor: [B, out_dim]
    """
    batch_size = len(guidance_list)
    dense = torch.zeros(batch_size, out_dim, dtype=torch.float32)

    for row, item in enumerate(guidance_list):
        hop_obj = item[hop_key]
        local_ids = hop_obj["local_ids"]
        weights = hop_obj["weights"]
        if local_ids.numel() > 0:
            dense[row, local_ids.long()] = weights.float()

    return dense


def _infer_hop_input_dim(is_mashup_side: bool, hop: int, n_mashup: int, n_api: int) -> int:
    """Infer the input dimension of each hop guidance feature."""
    if is_mashup_side:
        return n_api if hop % 2 == 1 else n_mashup
    return n_mashup if hop % 2 == 1 else n_api


def info_nce_loss(x: torch.Tensor, y: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    """Compute a symmetric InfoNCE loss between two embedding sets."""
    if x.size(0) == 0:
        return x.new_tensor(0.0)

    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)

    logits = torch.matmul(x, y.t()) / temperature
    labels = torch.arange(x.size(0), device=x.device)

    loss_xy = F.cross_entropy(logits, labels)
    loss_yx = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_xy + loss_yx)


class MLPPredictor(nn.Module):
    """Edge scorer that predicts mashup-API matching logits from node embeddings."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def apply_edges(self, edges):
        """Produce a scalar score for each edge."""
        edge_input = torch.cat([edges.src["h"], edges.dst["h"]], dim=1)
        score = self.fc2(F.relu(self.fc1(edge_input))).squeeze(1)
        return {"score": score}

    def forward(self, g: dgl.DGLGraph, h: torch.Tensor) -> torch.Tensor:
        """Apply the predictor on all edges of the graph."""
        with g.local_scope():
            g.ndata["h"] = h
            g.apply_edges(self.apply_edges)
            return g.edata["score"]


class LightGCNEncoder(nn.Module):
    """LightGCN encoder over the mashup-API interaction graph."""

    def __init__(self, num_nodes: int, hidden_dim: int, num_layers: int):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.id_embedding = nn.Embedding(num_nodes, hidden_dim)
        nn.init.xavier_uniform_(self.id_embedding.weight)

    def forward(self, g: dgl.DGLGraph) -> torch.Tensor:
        """Return the averaged node embeddings over all LightGCN layers."""
        with g.local_scope():
            h0 = self.id_embedding.weight

            deg = g.in_degrees().float().clamp(min=1)
            norm = torch.pow(deg, -0.5).unsqueeze(1).to(h0.device)

            h = h0
            out = h0

            for _ in range(self.num_layers):
                g.ndata["h"] = h * norm
                g.update_all(fn.copy_u("h", "m"), fn.sum("m", "h"))
                h = g.ndata["h"] * norm
                out = out + h

            out = out / (self.num_layers + 1.0)
            return out


class ConditionalDenoiser(nn.Module):
    """
    Conditional denoiser with multi-hop guidance.

    The model combines noisy latent states with hop-wise condition features,
    and aggregates hop outputs with learnable hop weights.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        diffusion_steps: int,
        hops: Tuple[int, ...] = (1, 2, 3),
        dropout: float = 0.1,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.diffusion_steps = diffusion_steps
        self.hops = tuple(hops)
        self.num_hops = len(self.hops)

        self.time_embedding = nn.Embedding(diffusion_steps, hidden_dim)
        self.z_proj = nn.Linear(hidden_dim, hidden_dim)

        self.q_proj = nn.ModuleDict()
        self.k_proj = nn.ModuleDict()
        self.v_proj = nn.ModuleDict()
        self.o_proj = nn.ModuleDict()
        self.ffn = nn.ModuleDict()
        self.norm1 = nn.ModuleDict()
        self.norm2 = nn.ModuleDict()

        for hop in self.hops:
            key = str(hop)
            self.q_proj[key] = nn.Linear(hidden_dim, hidden_dim)
            self.k_proj[key] = nn.Linear(hidden_dim, hidden_dim)
            self.v_proj[key] = nn.Linear(hidden_dim, hidden_dim)
            self.o_proj[key] = nn.Linear(hidden_dim, hidden_dim)
            self.norm1[key] = nn.LayerNorm(hidden_dim)
            self.norm2[key] = nn.LayerNorm(hidden_dim)
            self.ffn[key] = nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim * 2),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Dropout(dropout),
            )

        self.hop_logits = nn.Parameter(torch.zeros(self.num_hops))

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Split embeddings into multi-head format."""
        num_nodes, _ = x.shape
        return x.view(num_nodes, self.num_heads, self.head_dim)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Merge multi-head tensors back to dense embeddings."""
        num_nodes, num_heads, head_dim = x.shape
        return x.reshape(num_nodes, num_heads * head_dim)

    def _single_hop_cross_attention(
        self,
        z_t: torch.Tensor,
        cond_h: torch.Tensor,
        t_emb: torch.Tensor,
        hop: int,
    ) -> torch.Tensor:
        """
        Apply cross-attention for a single hop condition.

        Args:
            z_t: noisy latent, shape [N, D]
            cond_h: hop condition embeddings, shape [N, D]
            t_emb: timestep embeddings, shape [N, D]
            hop: hop id
        Returns:
            hidden_h: refined embeddings, shape [N, D]
        """
        key = str(hop)

        direct = self.z_proj(z_t + t_emb)
        direct = self.norm1[key](direct)

        q = self._split_heads(self.q_proj[key](cond_h))
        k = self._split_heads(self.k_proj[key](direct))
        v = self._split_heads(self.v_proj[key](direct))

        attn_score = (q * k).sum(dim=-1, keepdim=True) / math.sqrt(self.head_dim)
        attn_weight = torch.softmax(attn_score, dim=1)

        attn_out = attn_weight * v
        attn_out = self._merge_heads(attn_out)
        attn_out = self.o_proj[key](attn_out)

        hidden = z_t + attn_out
        hidden = self.norm2[key](hidden)
        hidden = hidden + self.ffn[key](torch.cat([hidden, cond_h, t_emb], dim=-1))
        return hidden

    def forward(self, z_t: torch.Tensor, cond_dict: Dict[int, torch.Tensor], t: torch.Tensor) -> torch.Tensor:
        """
        Predict the clean latent representation from noisy input.

        Args:
            z_t: noisy latent, shape [N, D]
            cond_dict: per-hop condition embeddings
            t: diffusion step per node, shape [N]
        Returns:
            Predicted clean latent x0, shape [N, D]
        """
        t_emb = self.time_embedding(t)
        alpha = torch.softmax(self.hop_logits, dim=0)

        hop_outputs = []
        for idx, hop in enumerate(self.hops):
            cond_h = cond_dict[hop]
            hidden_h = self._single_hop_cross_attention(z_t, cond_h, t_emb, hop)
            hop_outputs.append(alpha[idx] * hidden_h)

        return torch.stack(hop_outputs, dim=0).sum(dim=0)


class GatedFusion(nn.Module):
    """
    CF-dominant gated residual fusion.

    The collaborative representation is the backbone, and the denoised
    auxiliary representation is injected as a gated residual.
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.side_proj = nn.Linear(hidden_dim, hidden_dim)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, h_cf: torch.Tensor, h_aux: torch.Tensor) -> torch.Tensor:
        """Fuse collaborative and auxiliary embeddings."""
        side = self.dropout(self.side_proj(h_aux))
        gate = self.gate(torch.cat([h_cf, h_aux], dim=-1))
        return h_cf + gate * side


class Recommender(nn.Module):
    """
    Collaborative-guided diffusion recommender for mashup-API matching.

    Forward returns:
        h_fused: [num_nodes, hidden_dim]
        loss_reg: scalar regularization loss
    """

    def __init__(self, n_params: Dict):
        super().__init__()

        self.n_mashup = int(n_params["n_mashup"])
        self.n_api = int(n_params["n_api"])
        self.num_nodes = self.n_mashup + self.n_api

        self.hidden_dim = int(n_params.get("hidden_dim", 64))
        self.num_layers = int(n_params.get("num_layers", 2))
        self.num_heads = int(n_params.get("num_heads", 4))
        self.diffusion_steps = int(n_params.get("diffusion_steps", 5))
        self.dropout = float(n_params.get("dropout", 0.1))

        self.lambda_diff = float(n_params.get("lambda_diff", 1.0))
        self.lambda_cl = float(n_params.get("lambda_cl", 0.1))
        self.temperature = float(n_params.get("temperature", 0.2))

        self.hops = tuple(n_params.get("hops", (1, 2, 3)))
        self.num_hops = len(self.hops)
        if self.num_hops == 0:
            raise ValueError("hops cannot be empty")

        self.cf_encoder = LightGCNEncoder(
            num_nodes=self.num_nodes,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
        )

        mashup_content = _load_tensor(n_params["mashup_content"], "mashup_content")
        api_content = _load_tensor(n_params["api_content"], "api_content")

        if mashup_content.size(0) != self.n_mashup:
            raise ValueError(
                f"mashup_content rows={mashup_content.size(0)} != n_mashup={self.n_mashup}"
            )
        if api_content.size(0) != self.n_api:
            raise ValueError(
                f"api_content rows={api_content.size(0)} != n_api={self.n_api}"
            )

        self.register_buffer("mashup_content_raw", mashup_content)
        self.register_buffer("api_content_raw", api_content)

        self.mashup_content_proj = nn.Sequential(
            nn.Linear(mashup_content.size(1), self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.api_content_proj = nn.Sequential(
            nn.Linear(api_content.size(1), self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        guidance_obj = n_params["guidance"]
        mashup_guidance = guidance_obj["mashup_guidance"]
        api_guidance = guidance_obj["api_guidance"]

        for hop in self.hops:
            mashup_dim = _infer_hop_input_dim(True, hop, self.n_mashup, self.n_api)
            api_dim = _infer_hop_input_dim(False, hop, self.n_mashup, self.n_api)

            mashup_dense = _build_dense_guidance(mashup_guidance, f"hop{hop}", mashup_dim)
            api_dense = _build_dense_guidance(api_guidance, f"hop{hop}", api_dim)

            self.register_buffer(f"m_h{hop}_dense", mashup_dense)
            self.register_buffer(f"a_h{hop}_dense", api_dense)

        self.m_cond_encoders = nn.ModuleDict()
        self.a_cond_encoders = nn.ModuleDict()

        for hop in self.hops:
            mashup_dim = _infer_hop_input_dim(True, hop, self.n_mashup, self.n_api)
            api_dim = _infer_hop_input_dim(False, hop, self.n_mashup, self.n_api)

            self.m_cond_encoders[str(hop)] = nn.Sequential(
                nn.Linear(mashup_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.SiLU(),
                nn.Dropout(self.dropout),
            )
            self.a_cond_encoders[str(hop)] = nn.Sequential(
                nn.Linear(api_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.SiLU(),
                nn.Dropout(self.dropout),
            )

        self.denoiser = ConditionalDenoiser(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            diffusion_steps=self.diffusion_steps,
            hops=self.hops,
            dropout=self.dropout,
        )

        beta_start = float(n_params.get("beta_start", 1e-4))
        beta_end = float(n_params.get("beta_end", 2e-2))
        betas = torch.linspace(beta_start, beta_end, self.diffusion_steps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)

        self.fusion = GatedFusion(hidden_dim=self.hidden_dim, dropout=self.dropout)

    def _get_content_latent(self) -> torch.Tensor:
        """Project raw content features into the auxiliary latent space."""
        mashup_z0 = self.mashup_content_proj(self.mashup_content_raw)
        api_z0 = self.api_content_proj(self.api_content_raw)
        return torch.cat([mashup_z0, api_z0], dim=0)

    def _build_condition_dict(self) -> Dict[int, torch.Tensor]:
        """
        Build per-hop condition embeddings for all nodes.

        Returns:
            cond_dict[h]: [num_nodes, hidden_dim]
        """
        cond_dict: Dict[int, torch.Tensor] = {}

        for hop in self.hops:
            mashup_dense = getattr(self, f"m_h{hop}_dense")
            api_dense = getattr(self, f"a_h{hop}_dense")

            mashup_h = self.m_cond_encoders[str(hop)](mashup_dense)
            api_h = self.a_cond_encoders[str(hop)](api_dense)

            cond_dict[hop] = torch.cat([mashup_h, api_h], dim=0)

        return cond_dict

    def _q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample the noisy latent x_t from x0 at diffusion step t."""
        alpha_bar_t = self.alpha_bars[t].unsqueeze(1)
        x_t = alpha_bar_t.sqrt() * x0 + (1.0 - alpha_bar_t).sqrt() * noise
        return x_t, alpha_bar_t

    def _diffusion_forward(self, x0: torch.Tensor, cond_dict: Dict[int, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run one diffusion training step for all nodes.

        The denoiser predicts x0 directly instead of predicting noise.
        """
        device = x0.device
        num_nodes = x0.size(0)

        t = torch.randint(
            low=0,
            high=self.diffusion_steps,
            size=(num_nodes,),
            device=device,
        )
        noise = torch.randn_like(x0)

        x_t, _ = self._q_sample(x0, t, noise)
        x0_hat = self.denoiser(x_t, cond_dict, t)
        diff_loss = F.mse_loss(x0_hat, x0)

        return x0_hat, diff_loss

    def _alignment_loss(self, h_cf: torch.Tensor, h_aux: torch.Tensor) -> torch.Tensor:
        """Align CF embeddings with denoised auxiliary embeddings by node type."""
        mashup_cf = h_cf[: self.n_mashup]
        api_cf = h_cf[self.n_mashup :]

        mashup_aux = h_aux[: self.n_mashup]
        api_aux = h_aux[self.n_mashup :]

        loss_m = info_nce_loss(mashup_cf, mashup_aux, self.temperature)
        loss_a = info_nce_loss(api_cf, api_aux, self.temperature)
        return 0.5 * (loss_m + loss_a)

    def forward(self, train_g: dgl.DGLGraph) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            h_fused: [num_nodes, hidden_dim]
            loss_reg: scalar
        """
        h_cf = self.cf_encoder(train_g)
        x0 = self._get_content_latent()
        cond_dict = self._build_condition_dict()
        h_aux, diff_loss = self._diffusion_forward(x0, cond_dict)
        h_fused = self.fusion(h_cf, h_aux)
        cl_loss = self._alignment_loss(h_cf, h_aux)
        loss_reg = self.lambda_diff * diff_loss + self.lambda_cl * cl_loss
        return h_fused, loss_reg


def compute_loss(
    pos_score: torch.Tensor,
    neg_score: torch.Tensor,
    loss_reg: torch.Tensor,
    h: torch.Tensor,
    n_mashup: int,
    data_args,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the total training loss.

    Returns:
        total_loss: BCE + regularization losses
        ranking_loss_only: BCE + embedding L2
    """
    scores = torch.cat([pos_score, neg_score], dim=0)
    labels = torch.cat(
        [torch.ones_like(pos_score), torch.zeros_like(neg_score)],
        dim=0,
    )

    decay = data_args.l2
    bce_loss = nn.BCEWithLogitsLoss()(scores, labels)

    mashup_emb = h[:n_mashup, :]
    api_emb = h[n_mashup:, :]
    regularizer = (torch.norm(mashup_emb) ** 2 + torch.norm(api_emb) ** 2) / 2.0
    emb_loss = decay * regularizer / max(n_mashup, 1)

    total_loss = bce_loss + 0.1 * loss_reg + emb_loss
    ranking_loss_only = bce_loss + emb_loss
    return total_loss, ranking_loss_only
