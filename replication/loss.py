"""WFDENet training loss (paper Eq. 12-13).

Ported from mmseg/models/losses/loss.py in the authors' release, with the
hyperparameters from configs/_base_/models/wfdenet_b1.py:
    loss_decode=dict(type='Loss', loss_name='bdice', use_sigmoid=True,
                     loss_weight=1.0, eps=1e-5)

    L = Dice(P_sd, G) + lambda * Dice(P_lfb, G) + lambda * Dice(P_hfb, G),  lambda = 0.5

Note the Dice here uses a *squared* denominator (Milletari), not the linear
|P| + |G| form used by smp's DiceLoss in losses.py.
"""

import torch
import torch.nn as nn


def binary_dice_loss(logits: torch.Tensor, target: torch.Tensor,
                     eps: float = 1e-5) -> torch.Tensor:
    """Sigmoid Dice averaged over classes then over the batch.

    logits, target: [B, C, H, W]. Each channel is an independent binary mask
    (lesions overlap, so this is multilabel, not softmax over classes).
    """
    pred = torch.sigmoid(logits)
    num_classes = pred.shape[1]

    loss = 0.0
    for i in range(num_classes):
        p = pred[:, i].flatten(1)
        t = target[:, i].flatten(1).float()

        a = torch.sum(p * t, 1)
        b = torch.sum(p * p, 1)
        c = torch.sum(t * t, 1)
        loss = loss + (1 - (2 * a + eps) / (b + c + eps))

    return (loss / num_classes).mean()


class WFDENetLoss(nn.Module):
    """Main supervision on the SD output plus auxiliary supervision on the
    lowest-level LFB and HFB outputs (§3.6)."""

    def __init__(self, aux_weight: float = 0.5, eps: float = 1e-5):
        super().__init__()
        self.aux_weight = aux_weight
        self.eps = eps

    def forward(self, outputs, target: torch.Tensor):
        if not isinstance(outputs, (tuple, list)):
            outputs = (outputs,)

        main_loss = binary_dice_loss(outputs[0], target, self.eps)
        total = main_loss
        aux_losses = []
        for aux in outputs[1:]:
            aux_loss = binary_dice_loss(aux, target, self.eps)
            aux_losses.append(aux_loss)
            total = total + self.aux_weight * aux_loss

        return total, {
            'loss_main': main_loss.detach(),
            **{f'loss_aux{i}': l.detach() for i, l in enumerate(aux_losses)},
        }
