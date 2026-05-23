import math
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as functional


class Attention(nn.Module):
    """
    Causal multi-head attention layer following the statistical induction heads paper.

    No MLP, no layer norm.
    Relative position bias is passed in from outside and shared across layers.

    When num_heads=1 this recovers the original single-head formulation.
    When num_heads>1, swapping heads gives an equivalent model, inducing
    a permutation symmetry in the parameter space.

    Args:
        d_model: internal dimension d
        num_heads: number of attention heads (must divide d_model)
    """

    def __init__(self, d_model: int, num_heads: int = 1):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Q, K, V projections and output projection (all d_model × d_model)
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        rel_pos_bias: torch.Tensor,
        return_attn_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x:             (batch, seq_len, d_model)
            rel_pos_bias:  (seq_len, seq_len) shared relative position bias
            return_attn_weights: if True also return attention weights

        Returns:
            output of shape (batch, seq_len, d_model)
            optionally attention weights of shape (batch, num_heads, seq_len, seq_len)
        """
        batch, seq_len, _ = x.shape
        H = self.num_heads
        D = self.head_dim

        q = self.W_q(x).view(batch, seq_len, H, D).transpose(1, 2)  # (B, H, S, D)
        k = self.W_k(x).view(batch, seq_len, H, D).transpose(1, 2)
        v = self.W_v(x).view(batch, seq_len, H, D).transpose(1, 2)

        # attention scores: (B, H, S, S)
        scale = math.sqrt(D)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale

        # add shared relative position bias (broadcast over batch & heads)
        scores = scores + rel_pos_bias.unsqueeze(0).unsqueeze(0)

        # causal mask
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=x.device),
            diagonal=1,
        )
        scores = scores + causal_mask

        attn_weights = functional.softmax(scores, dim=-1)  # (B, H, S, S)

        out = torch.matmul(attn_weights, v)  # (B, H, S, D)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        out = self.W_o(out)

        # residual connection
        out = x + out

        if return_attn_weights:
            return out, attn_weights
        return out


class MarkovTransformer(nn.Module):
    """
    Two layer attention-only transformer following the statistical induction
    heads paper (Edelman et al. 2024).

    Architecture:
        - One-hot input embeddings e_x = delta_x (no learned token embedding)
        - Linear projection P: vocab_size -> d_model where d_model = 3 * vocab_size
        - Single shared relative position bias vector v of length max_len
          shared across both layers (Shaw et al. 2018)
        - Two AttentionLayers, each with a single head and no MLP
        - Linear unembedding head to vocab_size

    Args:
        vocab_size: number of states k in the Markov chain
        max_len:    maximum sequence length
        num_heads:  attention heads per layer (default 1 = original model).
                    num_heads>1 introduces a head-permutation symmetry.
    """

    def __init__(
        self, vocab_size: int = 5, max_len: int = 64, d_model=None, num_heads: int = 1
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.max_len = max_len
        self.num_heads = num_heads

        if not d_model:
            # internal dimension d = 3k following the paper
            self.d_model = 3 * vocab_size
        else:
            self.d_model = d_model

        # linear projection from one-hot (vocab_size) to d_model
        # P in R^{vocab_size x d_model}
        self.input_proj = nn.Linear(vocab_size, self.d_model, bias=False)

        # single shared relative position bias vector
        # v in R^{max_len}, one scalar per relative distance
        # shared across both layers and the single head in each layer
        self.rel_pos_bias = nn.Embedding(max_len, 1)

        # two attention-only layers
        self.layers = nn.ModuleList(
            [
                Attention(d_model=self.d_model, num_heads=self.num_heads)
                for _ in range(2)
            ]
        )

        # linear unembedding head
        self.head = nn.Linear(self.d_model, vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _compute_rel_pos_bias(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Compute the (seq_len, seq_len) relative position bias matrix.
        Entry (i, j) is the bias for position i attending to position j.
        Distance is i - j, clipped to [0, max_len - 1].

        Returns:
            rel_bias: (seq_len, seq_len)
        """
        positions = torch.arange(seq_len, device=device)
        distances = (positions.unsqueeze(1) - positions.unsqueeze(0)).clamp(
            min=0, max=self.max_len - 1
        )
        return self.rel_pos_bias(distances).squeeze(-1)

    def forward(
        self,
        x: torch.Tensor,
        return_intermediates: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict]:
        """
        Args:
            x:                    integer token sequence (batch, seq_len)
            return_intermediates: if True return intermediate activations

        Returns:
            logits of shape (batch, seq_len, vocab_size)
            if return_intermediates: also returns dict with
                residual_stream: list of (batch, seq_len, d_model) tensors
                                 one per layer including input and output
                attn_weights:    list of (batch, seq_len, seq_len) tensors
                                 one per layer
        """
        _, seq_len = x.shape
        intermediates = {"residual_stream": [], "attn_weights": []}

        # one-hot encode: (batch, seq_len, vocab_size)
        x_one_hot = functional.one_hot(x, num_classes=self.vocab_size).float()

        # project to d_model: (batch, seq_len, d_model)
        h = self.input_proj(x_one_hot)

        if return_intermediates:
            intermediates["residual_stream"].append(h.detach())

        # precompute shared relative position bias once
        rel_pos_bias = self._compute_rel_pos_bias(seq_len, x.device)

        # two attention layers
        for layer in self.layers:
            if return_intermediates:
                h, attn_weights = layer(h, rel_pos_bias, return_attn_weights=True)
                intermediates["attn_weights"].append(attn_weights.detach())
                intermediates["residual_stream"].append(h.detach())
            else:
                h = layer(h, rel_pos_bias)

        # unembedding to logits
        logits = self.head(h)

        if return_intermediates:
            return logits, intermediates
        return logits

    def predict_probs(self, x: torch.Tensor) -> torch.Tensor:
        logits = cast(torch.Tensor, self.forward(x, return_intermediates=False))
        return logits.softmax(dim=-1)

    def recover_transition_matrix(
        self,
        n_samples: int = 200,
        context_len: int = 10,
        device: str = "cpu",
    ) -> torch.Tensor:
        """
        Recover the learned transition matrix by querying the network
        for each possible current state.

        For each state i, constructs n_samples random contexts ending
        in state i, passes through the network, and averages the output
        distribution at the last position to get T_hat[i, :].

        Returns:
            T_hat: (vocab_size, vocab_size) estimated transition matrix
        """
        self.eval()
        t_hat = torch.zeros(self.vocab_size, self.vocab_size, device=device)

        with torch.no_grad():
            for state in range(self.vocab_size):
                contexts = torch.randint(
                    0, self.vocab_size, (n_samples, context_len), device=device
                )
                query = torch.full((n_samples, 1), state, device=device)
                sequences = torch.cat([contexts, query], dim=1)
                probs = self.predict_probs(sequences)
                t_hat[state] = probs[:, -1, :].mean(dim=0)

            return t_hat

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
