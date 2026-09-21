"""GDE-CFM components added on top of the official iTransformer repository."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleVelocity(nn.Module):
    def __init__(self, scales=(1, 2, 4), score_hidden=16):
        super().__init__()
        self.scales = tuple(scales)
        self.scorer = nn.Sequential(
            nn.Linear(1, score_hidden),
            nn.SiLU(),
            nn.Linear(score_hidden, 1),
        )

    def forward(self, x):
        """x: [B, L, N], returns velocity with the same shape."""
        candidates = []
        for scale in self.scales:
            diff = (x[:, scale:, :] - x[:, :-scale, :]) / float(scale)
            diff = F.pad(diff, (0, 0, scale, 0))
            candidates.append(diff)
        stack = torch.stack(candidates, dim=1)  # [B, K, L, N]
        scores = self.scorer(stack.unsqueeze(-1)).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        return (weights * stack).sum(dim=1), weights


class VelocityEncoder(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        hidden = max(32, d_model // 2)
        self.network = nn.Sequential(
            nn.Conv1d(1, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(hidden, d_model, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, velocity):
        batch, length, channels = velocity.shape
        series = velocity.transpose(1, 2).reshape(batch * channels, 1, length)
        encoded = self.network(series).squeeze(-1)
        return encoded.reshape(batch, channels, -1)


def sinusoidal_time_embedding(tau, d_model):
    half = d_model // 2
    frequencies = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, device=tau.device, dtype=tau.dtype)
        / max(half - 1, 1)
    )
    angles = tau[:, None] * frequencies[None, :]
    embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    if embedding.shape[-1] < d_model:
        embedding = F.pad(embedding, (0, d_model - embedding.shape[-1]))
    return embedding


class FlowBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.norm_self = nn.LayerNorm(d_model)
        self.self_attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm_cross = nn.LayerNorm(d_model)
        self.cross_attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm_mlp = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, state, condition, time_embedding):
        query = self.norm_self(state)
        state = state + self.self_attention(
            query, query, query, need_weights=False
        )[0]
        query = self.norm_cross(state)
        state = state + self.cross_attention(
            query, condition, condition, need_weights=False
        )[0]
        state = state + self.mlp(
            self.norm_mlp(state) + time_embedding[:, None, :]
        )
        return state


class ConditionalVectorField(nn.Module):
    def __init__(
        self,
        pred_len,
        d_model,
        n_heads,
        n_layers,
        d_ff,
        dropout,
    ):
        super().__init__()
        self.d_model = d_model
        self.future_embedding = nn.Linear(pred_len, d_model)
        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.blocks = nn.ModuleList(
            [
                FlowBlock(d_model, n_heads, d_ff, dropout)
                for _ in range(n_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, pred_len)

    def forward(self, y_tau, tau, condition):
        state = self.future_embedding(y_tau.transpose(1, 2))
        time_embedding = self.time_mlp(
            sinusoidal_time_embedding(tau, self.d_model)
        )
        for block in self.blocks:
            state = block(state, condition, time_embedding)
        return self.output_projection(self.output_norm(state)).transpose(1, 2)


class GDECFM(nn.Module):
    def __init__(self, configs):
        super().__init__()
        scales = tuple(int(value) for value in configs.gde_scales.split(','))
        self.probe_times = tuple(
            float(value) for value in configs.gde_probe_times.split(',')
        )
        self.pred_len = configs.pred_len
        self.velocity_builder = MultiScaleVelocity(scales=scales)
        self.velocity_encoder = VelocityEncoder(configs.d_model)
        self.velocity_gate = nn.Linear(2 * configs.d_model, configs.d_model)
        self.vector_field = ConditionalVectorField(
            pred_len=configs.pred_len,
            d_model=configs.d_model,
            n_heads=configs.n_heads,
            n_layers=configs.gde_flow_layers,
            d_ff=configs.gde_flow_d_ff,
            dropout=configs.dropout,
        )
        self.gde_projector = nn.Sequential(
            nn.Linear(configs.pred_len, configs.d_model),
            nn.SiLU(),
            nn.Linear(configs.d_model, configs.d_model),
        )
        self.probe_scorer = nn.Linear(configs.d_model, 1)
        self.backbone_gate = nn.Linear(2 * configs.d_model, configs.d_model)

    def condition(self, normalized_history, backbone_hidden):
        velocity, scale_weights = self.velocity_builder(normalized_history)
        velocity_hidden = self.velocity_encoder(velocity)
        velocity_gate = torch.sigmoid(
            self.velocity_gate(
                torch.cat([backbone_hidden, velocity_hidden], dim=-1)
            )
        )
        condition = backbone_hidden + velocity_gate * velocity_hidden
        return condition, velocity_gate, scale_weights

    def cfm_loss(self, normalized_future, condition):
        y0 = torch.randn_like(normalized_future)
        tau = torch.rand(
            normalized_future.shape[0],
            device=normalized_future.device,
            dtype=normalized_future.dtype,
        )
        y_tau = (
            (1.0 - tau[:, None, None]) * y0
            + tau[:, None, None] * normalized_future
        )
        target = normalized_future - y0
        prediction = self.vector_field(y_tau, tau, condition)
        return F.mse_loss(prediction, target)

    def extract(self, condition):
        batch, channels, _ = condition.shape
        zero = condition.new_zeros((batch, self.pred_len, channels))
        projections = []
        for value in self.probe_times:
            tau = condition.new_full((batch,), value)
            direction = self.vector_field(zero, tau, condition)
            projections.append(
                self.gde_projector(direction.transpose(1, 2))
            )
        stack = torch.stack(projections, dim=1)  # [B, M, N, d_model]
        scores = self.probe_scorer(stack).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        gde = (weights.unsqueeze(-1) * stack).sum(dim=1)
        return gde, weights

    def enhance(self, backbone_hidden, gde):
        gate = torch.sigmoid(
            self.backbone_gate(torch.cat([backbone_hidden, gde], dim=-1))
        )
        return backbone_hidden + gate * gde, gate

