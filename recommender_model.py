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
            score_matrix[mapped_u[int(s)], api_id] = float(score_np[i])

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
    """Eq. (19): CF anchors, cross-view positives, both-view negatives."""
    if x.shape != y.shape or temperature <= 0:
        raise ValueError("Views must have matching shapes and temperature must be positive")
    if x.size(0) == 0:
        return x.sum() * 0.0
    x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
    cross = x @ y.t() / temperature
    same = x @ x.t() / temperature
    same = same.masked_fill(torch.eye(x.size(0), device=x.device, dtype=torch.bool), -torch.inf)
    denominator = torch.logsumexp(torch.cat([cross, same], dim=1), dim=1)
    return (denominator - cross.diagonal()).mean()


class DotProductPredictor(nn.Module):
    """Inner-product edge scores, Eq. (22)."""

    def forward(self, g: dgl.DGLGraph, h: torch.Tensor) -> torch.Tensor:
        src, dst = g.edges()
        return (h[src] * h[dst]).sum(dim=-1)


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
    """Cross-attention over connectivity orders and x0 prediction, Eqs. (14)-(15)."""

    def __init__(self, hidden_dim: int, diffusion_steps: int,
                 hops: Tuple[int, ...] = (1, 2, 3), dropout: float = 0.1):
        super().__init__()
        self.hops = tuple(hops)
        self.time_embedding = nn.Embedding(diffusion_steps, hidden_dim)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, z_t: torch.Tensor, cond_dict: Dict[int, torch.Tensor],
                t: torch.Tensor) -> torch.Tensor:
        tokens = torch.stack([cond_dict[hop] for hop in self.hops], dim=1)
        query = self.q_proj(z_t).unsqueeze(1)
        keys, values = self.k_proj(tokens), self.v_proj(tokens)
        weights = torch.softmax((query * keys).sum(-1) / math.sqrt(z_t.size(-1)), dim=1)
        context = (weights.unsqueeze(-1) * values).sum(dim=1)
        return self.output_mlp(torch.cat([z_t, context, self.time_embedding(t)], dim=-1))


class GatedFusion(nn.Module):
    """CF base plus gated calibrated content, Eq. (21)."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, h_cf: torch.Tensor, h_aux: torch.Tensor) -> torch.Tensor:
        gamma = torch.sigmoid(self.gate(torch.cat([h_cf, h_aux], dim=-1)))
        return h_cf + gamma * h_aux


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

        # Node-type-specific query/key/value and prediction parameters.
        self.m_denoiser = ConditionalDenoiser(self.hidden_dim, self.diffusion_steps, self.hops, self.dropout)
        self.a_denoiser = ConditionalDenoiser(self.hidden_dim, self.diffusion_steps, self.hops, self.dropout)

        beta_start = float(n_params.get("beta_start", 1e-4))
        beta_end = float(n_params.get("beta_end", 2e-2))
        if self.diffusion_steps < 1 or not (0 < beta_start <= beta_end < 1):
            raise ValueError("Require T >= 1 and 0 < beta_start <= beta_end < 1")
        betas = torch.linspace(beta_start, beta_end, self.diffusion_steps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)

        self.m_fusion = GatedFusion(self.hidden_dim)
        self.a_fusion = GatedFusion(self.hidden_dim)

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

    def _denoise(self, x_t: torch.Tensor, cond_dict: Dict[int, torch.Tensor],
                 t: torch.Tensor) -> torch.Tensor:
        split = self.n_mashup
        return torch.cat([
            self.m_denoiser(x_t[:split], {h: c[:split] for h, c in cond_dict.items()}, t[:split]),
            self.a_denoiser(x_t[split:], {h: c[split:] for h, c in cond_dict.items()}, t[split:]),
        ], dim=0)

    def _diffusion_forward(self, x0: torch.Tensor, cond_dict: Dict[int, torch.Tensor]) -> torch.Tensor:
        """Uniform single-step x0 prediction for Eq. (23) only; indices are t-1."""
        t = torch.randint(self.diffusion_steps, (x0.size(0),), device=x0.device)
        x_t, _ = self._q_sample(x0, t, torch.randn_like(x0))
        error = (self._denoise(x_t, cond_dict, t) - x0).square().sum(dim=-1)
        return error[:self.n_mashup].mean() + error[self.n_mashup:].mean()

    def _reverse_diffusion(self, x0: torch.Tensor, cond_dict: Dict[int, torch.Tensor],
                           noise: torch.Tensor = None) -> torch.Tensor:
        """Eqs. (13), (17)-(18): q(x_T|x0), then T reverse means without added noise."""
        t = torch.full((x0.size(0),), self.diffusion_steps - 1, device=x0.device, dtype=torch.long)
        x_t, _ = self._q_sample(x0, t, torch.randn_like(x0) if noise is None else noise)
        for step in range(self.diffusion_steps - 1, -1, -1):
            t = torch.full_like(t, step)
            prediction = self._denoise(x_t, cond_dict, t)
            if step == 0:  # alpha_bar_0 = 1: the last mean is the x0 prediction.
                x_t = prediction
            else:
                previous = self.alpha_bars[step - 1]
                denominator = 1.0 - self.alpha_bars[step]
                x_t = (self.alphas[step].sqrt() * (1.0 - previous) * x_t
                       + previous.sqrt() * self.betas[step] * prediction) / denominator
        return x_t

    def _alignment_loss(self, h_cf: torch.Tensor, h_aux: torch.Tensor) -> torch.Tensor:
        """Align CF embeddings with denoised auxiliary embeddings by node type."""
        mashup_cf = h_cf[: self.n_mashup]
        api_cf = h_cf[self.n_mashup :]

        mashup_aux = h_aux[: self.n_mashup]
        api_aux = h_aux[self.n_mashup :]

        loss_m = info_nce_loss(mashup_cf, mashup_aux, self.temperature)
        loss_a = info_nce_loss(api_cf, api_aux, self.temperature)
        return loss_m + loss_a

    def forward(self, train_g: dgl.DGLGraph) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            h_fused: [num_nodes, hidden_dim]
            loss_reg: scalar
        """
        h_cf = self.cf_encoder(train_g)
        x0 = self._get_content_latent()
        cond_dict = self._build_condition_dict()
        h_aux = self._reverse_diffusion(x0, cond_dict)
        split = self.n_mashup
        h_fused = torch.cat([
            self.m_fusion(h_cf[:split], h_aux[:split]),
            self.a_fusion(h_cf[split:], h_aux[split:]),
        ], dim=0)
        loss_reg = x0.new_zeros(())
        if self.training:
            diff_loss = self._diffusion_forward(x0, cond_dict)
            cl_loss = self._alignment_loss(h_cf, h_aux)
            loss_reg = self.lambda_diff * diff_loss + self.lambda_cl * cl_loss
        return h_fused, loss_reg


def compute_loss(pos_score: torch.Tensor, neg_score: torch.Tensor,
                 loss_reg: torch.Tensor, parameters, l2_weight: float = 1e-4
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Eqs. (24)-(25); score pairs must represent (m, positive API, negative API)."""
    if pos_score.shape != neg_score.shape or pos_score.numel() == 0:
        raise ValueError("BPR requires nonempty, aligned positive/negative scores")
    bpr_loss = -F.logsigmoid(pos_score - neg_score).sum()
    parameter_l2 = sum(p.square().sum() for p in parameters if p.requires_grad)
    ranking_loss = bpr_loss + l2_weight * parameter_l2
    return ranking_loss + loss_reg, ranking_loss
