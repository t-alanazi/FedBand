

import copy
import numpy as np
import torch
import torch.nn as nn



class ServerAggregator(nn.Module):
    """
    FedBand / FedAvg-style server aggregator.

    Here we implement:
      - aggregation weighted by validation loss
        (higher loss ⇒ higher weight, as in the paper)
    """

    def __init__(self, global_model, device=None):
        super().__init__()
        self.global_model = global_model
        if device is None:
            device = next(global_model.parameters()).device
        self.device = device

    def aggregate(self, client_updates, val_losses):
        """
        Args:
            client_updates: dict[cid] -> state_dict (full dense models)
            val_losses:     dict[cid] -> float validation loss
        """
        global_dict = self.global_model.state_dict()

        selected_cids = list(client_updates.keys())
        losses = []
        for cid in selected_cids:
            v = float(val_losses.get(cid, 0.0))
            if not np.isfinite(v) or v < 0:
                v = 0.0
            losses.append(v)

        losses_t = torch.tensor(losses, dtype=torch.float32, device=self.device)

        # higher loss ⇒ larger weight; normalize to sum 1
        eps = 1e-8
        losses_t = losses_t + eps
        if losses_t.sum() <= 0:
            weights = torch.full_like(losses_t, 1.0 / len(selected_cids))
        else:
            weights = losses_t / losses_t.sum()

        for key in global_dict:
            stacked = torch.stack(
                [client_updates[cid][key].float().to(self.device)
                 for cid in selected_cids],
                dim=0,
            )
            w_shape = [weights.shape[0]] + [1] * (stacked.dim() - 1)
            weighted_sum = torch.sum(stacked * weights.view(w_shape), dim=0)
            global_dict[key] = weighted_sum

        self.global_model.load_state_dict(global_dict)

