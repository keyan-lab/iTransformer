"""Official iTransformer backbone with the proposed GDE-CFM adapter."""

import torch
import torch.nn as nn

from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
from model.gde_cfm_modules import GDECFM


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm

        # The following backbone is intentionally identical to official
        # model/iTransformer.py.
        self.enc_embedding = DataEmbedding_inverted(
            configs.seq_len,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout,
        )
        self.class_strategy = configs.class_strategy
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=configs.output_attention,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

        # New module only; the official backbone above remains unchanged.
        self.gde_cfm = GDECFM(configs)

    def _encode_history(self, x_enc, x_mark_enc):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            centered = x_enc - means
            stdev = torch.sqrt(
                torch.var(centered, dim=1, keepdim=True, unbiased=False) + 1e-5
            )
            normalized = centered / stdev
        else:
            means = torch.zeros_like(x_enc[:, :1, :])
            stdev = torch.ones_like(x_enc[:, :1, :])
            normalized = x_enc

        channels = x_enc.shape[-1]
        enc_out = self.enc_embedding(normalized, x_mark_enc)
        enc_out, attention = self.encoder(enc_out, attn_mask=None)
        backbone_hidden = enc_out[:, :channels, :]
        return normalized, backbone_hidden, means, stdev, attention

    def forward(
        self,
        x_enc,
        x_mark_enc,
        x_dec,
        x_mark_dec,
        mask=None,
        y_future=None,
    ):
        (
            normalized_history,
            backbone_hidden,
            means,
            stdev,
            attention,
        ) = self._encode_history(x_enc, x_mark_enc)

        condition, velocity_gate, scale_weights = self.gde_cfm.condition(
            normalized_history, backbone_hidden
        )
        gde, probe_weights = self.gde_cfm.extract(condition)
        enhanced_hidden, backbone_gate = self.gde_cfm.enhance(
            backbone_hidden, gde
        )
        prediction = self.projector(enhanced_hidden).permute(0, 2, 1)

        if self.use_norm:
            prediction = prediction * stdev[:, 0, :].unsqueeze(1)
            prediction = prediction + means[:, 0, :].unsqueeze(1)

        auxiliary_loss = None
        if y_future is not None:
            normalized_future = (
                (y_future - means) / stdev if self.use_norm else y_future
            )
            auxiliary_loss = self.gde_cfm.cfm_loss(
                normalized_future, condition
            )

        return {
            'prediction': prediction[:, -self.pred_len:, :],
            'cfm_loss': auxiliary_loss,
            'attention': attention if self.output_attention else None,
            'gde': gde,
            'kinematic_condition': condition,
            'velocity_gate': velocity_gate,
            'backbone_gate': backbone_gate,
            'probe_weights': probe_weights,
            'scale_weights': scale_weights,
        }

