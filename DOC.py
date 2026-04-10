import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class DOC(nn.Module):
    def __init__(self, temperature=0.07, eps=1e-8):
        super().__init__()
        self.temperature = temperature
        self.eps = eps

        self.pseudo_memory: Dict[str, torch.Tensor] = dict()
        self.video_deviations: Dict[str, float] = dict()
        self.global_threshold = 0.0

    def _compute_metric(self, pred, target):
        pred_flat = pred.flatten(1)
        target_flat = target.flatten(1)
        e1 = (pred_flat * target_flat).sum(dim=1)
        e2 = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
        m1 = (2.0 * e1 + self.eps) / (e2 + self.eps)
        m2 = (e1 + self.eps) / (e2 - e1 + self.eps)
        return m1, m2

    def compute_deviation(self, current_pred, history_pred):
        current_bin = (current_pred > 0.5).float()
        history_bin = (history_pred > 0.5).float()
        m1, m2 = self._compute_metric(current_bin, history_bin)
        return (1.0 - (m1 + m2) / 2.0).mean().item()

    def compute_contrastive_loss(self, features, pseudo_labels):
        T, C, H, W = features.shape
        features = features.reshape(T, C, -1).permute(0, 2, 1)
        pseudo_labels = pseudo_labels.reshape(T, 1, -1).squeeze(1)
        features = F.normalize(features, dim=-1)

        total_loss = 0.0
        valid = 0

        for t in range(T):
            feat = features[t]
            label = pseudo_labels[t]
            fg_mask = label > 0.5
            bg_mask = label <= 0.5

            fg_feat = feat[fg_mask]
            bg_feat = feat[bg_mask]

            if fg_feat.numel() == 0 or bg_feat.numel() == 0:
                continue

            pos_sim = fg_feat @ fg_feat.T / self.temperature
            neg_sim = fg_feat @ bg_feat.T / self.temperature

            pos = torch.exp(pos_sim.diag())
            neg = torch.exp(neg_sim).sum(dim=1)
            loss = -torch.log(pos / (pos + neg + self.eps)).mean()

            total_loss += loss
            valid += 1

        return total_loss / valid if valid > 0 else torch.tensor(0.0, device=features.device)

    def forward(self, video_ids, current_preds, features, epoch):
        device = current_preds.device
        self.video_deviations.clear()

        if epoch == 0:
            for vid, pred in zip(video_ids, current_preds):
                self.pseudo_memory[vid] = pred.detach()
            return None, torch.tensor(0.0, device=device)

        supervised_labels = torch.zeros_like(current_preds)
        for idx, vid in enumerate(video_ids):
            curr_pred = current_preds[idx]
            hist_pred = self.pseudo_memory.get(vid, curr_pred.detach())

            dev = self.compute_deviation(curr_pred, hist_pred)
            self.video_deviations[vid] = dev

            if dev <= self.global_threshold:
                supervised_labels[idx] = curr_pred.detach()
                self.pseudo_memory[vid] = curr_pred.detach()
            else:
                supervised_labels[idx] = hist_pred.detach()

        loss = self.compute_contrastive_loss(features, supervised_labels)
        return supervised_labels, loss

    def update_threshold(self, epoch):
        if epoch == 1:
            self.global_threshold = 0.0
        elif epoch >= 2 and self.video_deviations:
            mean_dev = sum(self.video_deviations.values()) / len(self.video_deviations)
            self.global_threshold = mean_dev