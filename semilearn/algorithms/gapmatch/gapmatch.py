# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch

from semilearn.algorithms.hooks import PseudoLabelingHook
from semilearn.algorithms.utils import SSL_Argument, str2bool
from semilearn.core.algorithmbase import AlgorithmBase
from semilearn.core.utils import ALGORITHMS


@ALGORITHMS.register("gapmatch")
class GapMatch(AlgorithmBase):
    """
    GapMatch: Gradient-based Adversarial Perturbation for Semi-Supervised Learning
    """

    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger)
        self.init(
            T=args.T,
            p_cutoff=args.p_cutoff,
            hard_label=args.hard_label,
            alpha=args.alpha,
            epsilon=args.epsilon,
        )

    def init(self, T, p_cutoff, hard_label=True, alpha=1.0, epsilon=0.01):
        self.T = T
        self.p_cutoff = p_cutoff
        self.use_hard_label = hard_label
        self.alpha = alpha
        self.epsilon = epsilon

    def set_hooks(self):
        self.register_hook(PseudoLabelingHook(), "PseudoLabelingHook")
        super().set_hooks()

    def train_step(self, x_lb, y_lb, x_ulb_w, x_ulb_s):
        num_lb = y_lb.size(0)

        # ======================================================
        # 1. Forward (clean model)
        # ======================================================
        with self.amp_cm():
            outs_lb = self.model(x_lb)
            logits_lb = outs_lb["logits"]

            outs_ulb_w = self.model(x_ulb_w)
            logits_ulb_w = outs_ulb_w["logits"]

            outs_ulb_s = self.model(x_ulb_s)
            logits_ulb_s = outs_ulb_s["logits"]

            sup_loss = self.ce_loss(logits_lb, y_lb, reduction="mean")

        # ======================================================
        # 2. Pseudo-labels (DETACHED)
        # ======================================================
        with torch.no_grad():
            probs = torch.softmax(logits_ulb_w, dim=-1)
            max_probs, _ = torch.max(probs, dim=-1)
            mask = (max_probs >= self.p_cutoff).float()

            pseudo_label = self.call_hook(
                "gen_ulb_targets",
                "PseudoLabelingHook",
                logits=logits_ulb_w,
                use_hard_label=self.use_hard_label,
                T=self.T,
            )

        pseudo_label = pseudo_label.detach()
        mask = mask.detach()

        # ======================================================
        # 3. Vanilla unsupervised loss (L_unsup)
        # ======================================================
        with self.amp_cm():
            unsup_loss = self.consistency_loss(
                logits_ulb_s, pseudo_label, "ce", mask=mask
            )

        # ======================================================
        # 4. g1 = ∇θ L_unsup
        # ======================================================
        g1 = torch.autograd.grad(
            unsup_loss,
            self.model.parameters(),
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )

        # ======================================================
        # 5. Apply adversarial perturbation
        # ======================================================
        param_backup = {}
        grad_norm = 0.0

        for (name, param), grad in zip(self.model.named_parameters(), g1):
            if grad is not None:
                param_backup[name] = param.data.clone()
                grad_norm += grad.norm(2).item() ** 2

        grad_norm = grad_norm**0.5

        if grad_norm > 0:
            for (name, param), grad in zip(self.model.named_parameters(), g1):
                if grad is not None:
                    param.data.add_(self.epsilon * grad / grad_norm)

        # ======================================================
        # 6. Forward (perturbed model)
        # ======================================================
        with self.amp_cm():
            logits_ulb_s_adv = self.model(x_ulb_s)["logits"]

            unsup_loss_adv = self.consistency_loss(
                logits_ulb_s_adv, pseudo_label, "ce", mask=mask
            )

        # ======================================================
        # 7. Restore parameters
        # ======================================================
        for name, param in self.model.named_parameters():
            if name in param_backup:
                param.data.copy_(param_backup[name])

        # ======================================================
        # 8. Final loss (single backward by SemiLearn)
        # ======================================================
        total_loss = sup_loss + self.lambda_u * (
            unsup_loss + self.alpha * unsup_loss_adv.detach()
        )

        # ======================================================
        # 9. Logging
        # ======================================================
        log_dict = self.process_log_dict(
            sup_loss=sup_loss.item(),
            unsup_loss=unsup_loss.item(),
            unsup_loss_adv=unsup_loss_adv.item(),
            total_loss=total_loss.item(),
            util_ratio=mask.mean().item(),
        )

        out_dict = self.process_out_dict(loss=total_loss)

        return out_dict, log_dict

    @staticmethod
    def get_argument():
        return [
            SSL_Argument("--hard_label", str2bool, True),
            SSL_Argument("--T", float, 0.5),
            SSL_Argument("--p_cutoff", float, 0.95),
            SSL_Argument("--alpha", float, 1.0),
            SSL_Argument("--epsilon", float, 0.01),
        ]
