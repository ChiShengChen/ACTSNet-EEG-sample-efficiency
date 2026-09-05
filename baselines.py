"""Reference EEGNet & ShallowConvNet baselines for the ACTSNet LOSO comparison.

Faithful PyTorch reimplementations of the two standard EEG CNNs (architectures
match braindecode / the original papers). Vendored rather than imported from
braindecode because braindecode 1.5's eager package __init__ pulls skorch/wfdb/…
and is fragile under torch 2.9 — these self-contained modules avoid that and are
fully reproducible.

  - EEGNet:        Lawhern et al. 2018, "EEGNet: a compact CNN for EEG-based BCIs"
  - ShallowConvNet: Schirrmeister et al. 2017, "Deep learning with CNNs for EEG decoding"

Input: (B, C, T). A Linear head sized from a dummy forward, so any C/T works.
"""

import torch
import torch.nn as nn


class _Square(nn.Module):
    def forward(self, x):
        return x * x


class _Log(nn.Module):
    def forward(self, x):
        return torch.log(torch.clamp(x, min=1e-6))


def _infer_flat_dim(feature_extractor, n_chans, n_times):
    with torch.no_grad():
        d = torch.zeros(1, 1, n_chans, n_times)
        return feature_extractor(d).reshape(1, -1).shape[1]


class EEGNet(nn.Module):
    """EEGNetv4. F1 temporal filters, D depth multiplier, F2 separable filters."""

    def __init__(self, n_chans, n_classes, n_times, F1=8, D=2, F2=16,
                 kernel_length=64, drop=0.25):
        super().__init__()
        self.features = nn.Sequential(
            # temporal conv
            nn.Conv2d(1, F1, (1, kernel_length), padding=(0, kernel_length // 2), bias=False),
            nn.BatchNorm2d(F1),
            # depthwise spatial conv over all channels
            nn.Conv2d(F1, F1 * D, (n_chans, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(drop),
            # separable conv
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(drop),
        )
        self.classifier = nn.Linear(_infer_flat_dim(self.features, n_chans, n_times), n_classes)

    def forward(self, x):                    # x: (B, C, T)
        x = x.unsqueeze(1)                   # (B, 1, C, T)
        x = self.features(x)
        return self.classifier(x.reshape(x.size(0), -1))


class ShallowConvNet(nn.Module):
    """ShallowConvNet: temporal + spatial conv, square -> avgpool -> log."""

    def __init__(self, n_chans, n_classes, n_times, n_filters=40,
                 temp_len=25, pool_len=75, pool_stride=15, drop=0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, n_filters, (1, temp_len), bias=False),          # temporal
            nn.Conv2d(n_filters, n_filters, (n_chans, 1), bias=False),    # spatial
            nn.BatchNorm2d(n_filters),
            _Square(),
            nn.AvgPool2d((1, pool_len), stride=(1, pool_stride)),
            _Log(),
            nn.Dropout(drop),
        )
        self.classifier = nn.Linear(_infer_flat_dim(self.features, n_chans, n_times), n_classes)

    def forward(self, x):                    # x: (B, C, T)
        x = x.unsqueeze(1)
        x = self.features(x)
        return self.classifier(x.reshape(x.size(0), -1))


class BigCNN(nn.Module):
    """Capacity-matched plain 1D-CNN control (~ACTSNet's parameter count, no metric-learning
    inductive bias): a deep conv stack + GAP + MLP softmax head. Used to test whether the
    low-data advantage comes from capacity rather than inductive bias — if a same-size plain
    CNN gets no low-data benefit, the advantage is attributable to ACTSNet's design.
    """

    def __init__(self, n_chans, n_classes, n_times, widths=(96, 160, 256), fc=320, k=7, drop=0.3):
        super().__init__()
        layers, c = [], n_chans
        for w in widths:
            layers += [nn.Conv1d(c, w, k, padding=k // 2), nn.BatchNorm1d(w), nn.ELU(),
                       nn.MaxPool1d(4), nn.Dropout(drop)]
            c = w
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                                  nn.Linear(widths[-1], fc), nn.ELU(), nn.Dropout(drop),
                                  nn.Linear(fc, n_classes))

    def forward(self, x):                     # x: (B, C, T)
        return self.head(self.features(x))


class EEGTransformer(nn.Module):
    """Compact modern transformer baseline: conv patch tokenizer -> positional embedding ->
    TransformerEncoder -> mean-pool -> linear head. A recent-architecture reference point."""

    def __init__(self, n_chans, n_classes, n_times, d_model=64, nhead=4, layers=3,
                 patch=32, drop=0.3):
        super().__init__()
        self.tok = nn.Conv1d(n_chans, d_model, patch, stride=patch)   # (B, d_model, T//patch)
        n_tok = n_times // patch
        self.pos = nn.Parameter(torch.randn(1, n_tok, d_model) * 0.02)
        enc = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model * 4,
                                         dropout=drop, batch_first=True, activation="gelu")
        self.tr = nn.TransformerEncoder(enc, layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Dropout(drop),
                                  nn.Linear(d_model, n_classes))

    def forward(self, x):                     # x: (B, C, T)
        z = self.tok(x).transpose(1, 2)       # (B, n_tok, d_model)
        z = z + self.pos[:, :z.size(1)]
        z = self.tr(z)
        return self.head(z.mean(dim=1))


BASELINES = {"eegnet": EEGNet, "shallowconv": ShallowConvNet, "bigcnn": BigCNN,
             "transformer": EEGTransformer}
