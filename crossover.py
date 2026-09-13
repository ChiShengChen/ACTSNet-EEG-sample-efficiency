"""2x2 factorial control: encoder architecture x classification head.

Reviewer AN (comment 2) on JMIR ms#107929: the manuscript attributes the low-data
advantage to *architecture*, but ACTSNet uses a prototypical head while the
size-matched control (BigCNN) uses cross-entropy -- architecture and objective
change together, so "the advantage comes from the loss" is equally plausible.

This module supplies the missing cells so all four combinations exist under one
identical protocol (same folds, seeds, support-set construction, model selection):

                      | prototypical head | softmax head
    ------------------+-------------------+--------------------------
    ACTSNet encoder   | full model        | H3 ablation (--head softmax)
    BigCNN encoder    | *this module*     | existing `bigcnn` baseline

`BigCNNProto` exposes exactly the interface run_loso.py expects of ACTSNet
(`encode`, `forward(x, support_x, support_labels)`) and reuses ACTSNet's own
`PrototypicalLearning` head unchanged, so the encoder is the only thing that differs.
"""
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# The ACTSNet architecture lives in a separate repository:
#   https://github.com/ChiShengChen/ACTSNetv1
# Set ACTSNET_V1 to a local checkout, or clone it next to this repository.
sys.path.insert(0, os.environ.get("ACTSNET_V1",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "ACTSNetv1")))
from actsnet.model import ACTSNet, PrototypicalLearning  # noqa: E402

from baselines import BigCNN  # noqa: E402


class BigCNNProto(nn.Module):
    """BigCNN's convolutional trunk + ACTSNet's attentional prototypical head.

    The trunk is taken verbatim from `baselines.BigCNN` (same widths/kernel/dropout,
    so capacity stays matched); only its softmax head is replaced by a projection to
    `prototype_dim` followed by the prototypical classifier.
    """

    def __init__(self, config):
        super().__init__()
        trunk = BigCNN(n_chans=config.n_channels, n_classes=config.n_classes,
                       n_times=getattr(config, "n_times", 1024))
        self.features = trunk.features                      # conv stack, verbatim
        width = self.features[-4].num_features              # BatchNorm1d of last block
        self.pool = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten())
        self.proj = nn.Sequential(nn.Linear(width, config.prototype_dim), nn.ELU(),
                                  nn.Dropout(config.dropout))
        nn.init.xavier_normal_(self.proj[0].weight)

        self.head = getattr(config, "head", "proto")
        if self.head == "softmax":
            self.classifier = nn.Linear(config.prototype_dim, config.n_classes)
            nn.init.xavier_normal_(self.classifier.weight)
        else:
            self.proto = PrototypicalLearning(
                embedding_dim=config.prototype_dim,
                n_classes=config.n_classes,
                latent_dim_u=config.latent_dim_u,
            )

    def encode(self, x):                       # (B, C, T) -> (B, prototype_dim)
        return self.proj(self.pool(self.features(x)))

    def forward(self, x, support_x=None, support_labels=None):
        query_emb = self.encode(x)
        if self.head == "softmax":
            return F.log_softmax(self.classifier(query_emb), dim=1)
        support_emb = query_emb if support_x is None else self.encode(support_x)
        return self.proto(query_emb, support_emb, support_labels)




class ACTSNetNoMultiScale(ACTSNet):
    """Editor comment 8: ablate the multi-scale branch.

    The manuscript ablates the Attentional Convolution branch (-> LSTM) and the
    prototypical head (-> softmax) but never the multi-scale branch, which is
    inherited unchanged from TapNet -- so the claim that the components are
    complementary is untested for that third component.

    Here the multi-scale branch is removed and the final projection is resized to
    take the sequence branch alone; everything else is inherited from ACTSNet.
    """

    def __init__(self, config):
        super().__init__(config)
        self.multi_scale = nn.Identity()      # drops its parameters from the optimiser
        self.final_fc = nn.Linear(config.prototype_dim, config.prototype_dim)
        nn.init.xavier_normal_(self.final_fc.weight)

    def encode(self, x):
        branch_in = self.channel_proj(x)
        branch_out = (self.seq_encoder(branch_in) if self.branch == "lstm"
                      else self.ac_encoder(branch_in))
        return self.final_fc(branch_out)


def build_model(config, encoder="actsnet", multiscale="on"):
    """Single place that maps the ablation flags to a model.

    encoder=actsnet, multiscale=on   -> ACTSNet (full model)
    encoder=actsnet, multiscale=off  -> multi-scale branch ablation (editor #8)
    encoder=bigcnn                   -> capacity-matched crossover (Reviewer AN #2)
    """
    if encoder == "bigcnn":
        if multiscale == "off":
            raise ValueError("--multiscale off 對 bigcnn encoder 無意義(它沒有該分支)")
        return BigCNNProto(config)
    return ACTSNet(config) if multiscale == "on" else ACTSNetNoMultiScale(config)
