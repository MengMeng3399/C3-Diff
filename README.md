# C3-Diff: Calibrating content via high-order connectivity-guided diffusion for cloud API recommendation
Research source code for C3-Diff. Processed graphs, content tensors, experiment splits, checkpoints, and a packaged execution environment are not bundled in this repository.

⭐ Overall framework of the C3-Diff model⭐ 

<img width="6469" height="4169" alt="flowchart5" src="https://github.com/user-attachments/assets/39064884-68f0-4980-a558-2a7410384fa4" />

Overall framework of the C3-Diff model. 

⭐ The HGA dataset refers to:[ https://github.com/528Lab/CAData](https://github.com/528Lab/CAData)⭐ 

⭐ The PWA dataset refers to: [https://github.com/kkfletch/API-Dataset](https://github.com/kkfletch/API-Dataset)⭐  


## Method correspondence

- `build_guidance_signals.py`: length-h walk counts that never revisit the target, normalized within each hop (Section 4.3). Third-hop neighbors can also be first-hop neighbors. Regenerate guidance tensors produced by the previous shortest-distance implementation.
- `recommender_model.py`: connectivity-order cross-attention (Eqs. 14-15), recursive reverse means (Eqs. 17-18), both-view contrastive negatives (Eq. 19), gated residual fusion (Eq. 21), and inner-product scores (Eq. 22).
- `train_recommender.py`: one unobserved API sampled per positive edge, preserving the Mashup and edge order for BPR (Eq. 24). The objective uses parameter L2 and the diffusion/contrastive weights directly (Eq. 25).

The random-timestep prediction is used only for diffusion reconstruction loss. Training and evaluation obtain calibrated content through the full reverse chain, initialized by noising the content at step T. Evaluation therefore still depends on the initial Gaussian draw; fixing the random seed controls that draw.

The revised architecture and objective are incompatible with checkpoints from the earlier MLP-scoring implementation. Source-level alignment with the manuscript does not by itself establish reproduction of the reported experimental results.
