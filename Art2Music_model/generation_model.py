import torch
import torch.nn as nn

# ---------------------- TextImage2Mel (改进版双向 LSTM) ----------------------
class TextImage2Mel(nn.Module):
    def __init__(self, feat_dim=512, hidden_dim=512, mel_dim=80, num_layers=4, max_len=896):
        super().__init__()
        self.feat_proj = nn.Linear(feat_dim, hidden_dim)
        self.pos_emb = nn.Parameter(torch.randn(1, max_len, hidden_dim))
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.out_proj = nn.Linear(hidden_dim * 2, mel_dim)

    def forward(self, fused_feat, out_len=896):
        if fused_feat.ndim == 1:
            fused_feat = fused_feat.unsqueeze(0)
        elif fused_feat.ndim == 3 and fused_feat.size(1) == 1:
            fused_feat = fused_feat.squeeze(1)

        B = fused_feat.size(0)
        x = self.feat_proj(fused_feat)
        x = x.unsqueeze(1).expand(B, out_len, -1)
        x = x + self.pos_emb[:, :out_len, :]
        x, _ = self.lstm(x)
        mel = self.out_proj(x)
        return mel  # [B, T, mel_dim]


class GatedResidualProjector(nn.Module):
    def __init__(self, feat_dim=1024, hidden_dim=512):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(feat_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        self.residual = nn.Linear(feat_dim, hidden_dim)

    def forward(self, text_feat, image_feat):
        concat = torch.cat([text_feat, image_feat], dim=-1)
        out = self.linear(concat)
        res = self.residual(text_feat)
        gate = self.gate(out)
        fused = gate * out + (1 - gate) * res
        return fused

