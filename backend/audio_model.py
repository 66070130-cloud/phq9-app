import torch
import torch.nn as nn


# ── Hyperparameters ตรงกับ depression_audio_only_v4.ipynb ────────
D_AUDIO_RAW = 768
D_MODEL     = 256
NHEAD       = 8
NUM_LAYERS  = 2
DROPOUT     = 0.3
NUM_CLASSES = 4   # 0=ไม่เลย, 1=บางวัน, 2=มากกว่าครึ่ง, 3=แทบทุกวัน
N_SEGMENTS  = 16


class AudioOnlyModel(nn.Module):
    """
    Architecture ตรงกับ depression_audio_only_v4.ipynb
    Input : (B, T, 768)  — Wav2Vec2 features, T = N_SEGMENTS = 16
    Output: (B, 4)       — [ไม่เลย, บางวัน, มากกว่าครึ่ง, แทบทุกวัน]
    """

    def __init__(
        self,
        d_in=D_AUDIO_RAW,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        num_classes=NUM_CLASSES,
    ):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_in, 512),     nn.BatchNorm1d(512),     nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, 384),      nn.BatchNorm1d(384),     nn.GELU(), nn.Dropout(dropout),
            nn.Linear(384, d_model),  nn.BatchNorm1d(d_model), nn.GELU(),
        )
        enc = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 768)
        x = self.proj(x.mean(dim=1))                          # (B, 256)
        x = self.transformer(x.unsqueeze(1)).squeeze(1)       # (B, 256)
        return self.classifier(x)                             # (B, 4)