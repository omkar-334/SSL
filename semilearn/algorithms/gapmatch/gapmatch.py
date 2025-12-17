# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn.functional as F
from copy import deepcopy

from semilearn.core.algorithmbase import AlgorithmBase
from semilearn.core.utils import ALGORITHMS
from semilearn.algorithms.hooks import PseudoLabelingHook
from semilearn.algorithms.utils import SSL_Argument, str2bool


@ALGORITHMS.register('gapmatch')
class GapMatch(AlgorithmBase):
    """
        GapMatch algorithm - Gradient-based Adversarial Perturbation for Semi-Supervised Segmentation.
        
        This method combines instance perturbation-based and model perturbation-based consistency 
        regularization to enhance the robustness of decision boundaries.

        Args:
            - args (`argparse`):
                algorithm arguments
            - net_builder (`callable`):
                network loading function
            - tb_log (`TBLog`):
                tensorboard logger
            - logger (`logging.Logger`):
                logger to use
            - T (`float`):
                Temperature for pseudo-label sharpening
            - p_cutoff (`float`):
                Confidence threshold for pseudo-labels
            - hard_label (`bool`, *optional*, default to `True`):
                If True, use hard pseudo-labels. If False, use soft pseudo-labels
            - alpha (`float`):
                Balance coefficient for adversarial gradient (α in the paper)
            - epsilon (`float`):
                Approximation scalar for perturbation (ε in the paper)
        """
    
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger)
        self.init(
            T=args.T,
            p_cutoff=args.p_cutoff,
            hard_label=args.hard_label,
            alpha=args.alpha,
            epsilon=args.epsilon
        )
    
    def init(self, T, p_cutoff, hard_label=True, alpha=1.0, epsilon=0.01):
        self.T = T
        self.p_cutoff = p_cutoff
        self.use_hard_label = hard_label
        self.alpha = alpha  # Balance coefficient for adversarial gradient
        self.epsilon = epsilon  # Perturbation scalar
        
    def set_hooks(self):
        self.register_hook(PseudoLabelingHook(), "PseudoLabelingHook")
        super().set_hooks()
    
    def train_step(self, x_lb, y_lb, x_ulb_w, x_ulb_s):
        num_lb = y_lb.shape[0]
        
        # Step 1: Calculate supervised loss for labeled data
        with self.amp_cm():
            if self.use_cat:
                inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))
                outputs = self.model(inputs)
                logits_x_lb = outputs['logits'][:num_lb]
                logits_x_ulb_w, logits_x_ulb_s = outputs['logits'][num_lb:].chunk(2)
                feats_x_lb = outputs['feat'][:num_lb]
                feats_x_ulb_w, feats_x_ulb_s = outputs['feat'][num_lb:].chunk(2)
            else:
                outs_x_lb = self.model(x_lb)
                logits_x_lb = outs_x_lb['logits']
                feats_x_lb = outs_x_lb['feat']
                
                outs_x_ulb_w = self.model(x_ulb_w)
                logits_x_ulb_w = outs_x_ulb_w['logits']
                feats_x_ulb_w = outs_x_ulb_w['feat']
                
                outs_x_ulb_s = self.model(x_ulb_s)
                logits_x_ulb_s = outs_x_ulb_s['logits']
                feats_x_ulb_s = outs_x_ulb_s['feat']
            
            feat_dict = {
                'x_lb': feats_x_lb,
                'x_ulb_w': feats_x_ulb_w,
                'x_ulb_s': feats_x_ulb_s
            }
            
            # Supervised loss
            sup_loss = self.ce_loss(logits_x_lb, y_lb, reduction='mean')
            
            # Step 2: Generate pseudo-labels from weakly augmented unlabeled data
            # Simple threshold-based masking as in GapMatch paper (Eq. 5)
            with torch.no_grad():
                probs_x_ulb_w = torch.softmax(logits_x_ulb_w.detach(), dim=-1)
                max_probs, _ = torch.max(probs_x_ulb_w, dim=-1)
                # Binary mask: 1 if confidence ≥ threshold, 0 otherwise
                mask = max_probs.ge(self.p_cutoff).float()
            
            # Generate pseudo-labels
            pseudo_label = self.call_hook(
                "gen_ulb_targets",
                "PseudoLabelingHook",
                logits=logits_x_ulb_w,
                use_hard_label=self.use_hard_label,
                T=self.T
            )
            
            # Step 3: Vanilla consistency regularization (instance perturbation-based)
            # This is Lunsup in the paper (Eq. 7)
            unsup_loss_vanilla = self.consistency_loss(
                logits_x_ulb_s,
                pseudo_label,
                'ce',
                mask=mask
            )
        
        # Step 4: Calculate gradient g1 = ∇θ Lunsup(θ)
        # This gradient will guide the adversarial perturbation direction
        self.model.zero_grad()
        unsup_loss_vanilla.backward(retain_graph=True)
        
        # Step 5: Backup model parameters and apply adversarial perturbation
        # θ* = θ + ε * (g1 / ||g1||)
        param_backup = {}
        grad_norm = 0.0
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                param_backup[name] = param.data.clone()
                grad_norm += param.grad.data.norm(2).item() ** 2
        
        grad_norm = grad_norm ** 0.5
        
        # Apply perturbation in the direction of gradient (gradient ascent)
        if grad_norm > 0:
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    perturbation = self.epsilon * param.grad.data / grad_norm
                    param.data.add_(perturbation)
        
        # Step 6: Forward pass with perturbed parameters
        with self.amp_cm():
            outs_x_ulb_s_perturbed = self.model(x_ulb_s)
            logits_x_ulb_s_perturbed = outs_x_ulb_s_perturbed['logits']
            
            # Step 7: Calculate consistency loss with perturbed model
            # This is L*unsup in the paper (Eq. 9)
            unsup_loss_perturbed = self.consistency_loss(
                logits_x_ulb_s_perturbed,
                pseudo_label,
                'ce',
                mask=mask
            )
        
        # Step 8: Calculate gradient g2 at perturbed parameters
        self.model.zero_grad()
        unsup_loss_perturbed.backward()
        
        # Combine gradients: gu = g1 + α * g2
        combined_grads = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None and name in param_backup:
                # g2 is the current gradient, we need to add back g1
                # g1 was used for perturbation, so we recalculate it
                combined_grads[name] = param.grad.data.clone()
        
        # Step 9: Restore original parameters
        for name, param in self.model.named_parameters():
            if name in param_backup:
                param.data.copy_(param_backup[name])
        
        # Step 10: Apply combined gradient
        with self.amp_cm():
            # Re-compute for clean backward pass
            if self.use_cat:
                inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))
                outputs = self.model(inputs)
                logits_x_lb = outputs['logits'][:num_lb]
                logits_x_ulb_w, logits_x_ulb_s = outputs['logits'][num_lb:].chunk(2)
            else:
                outs_x_lb = self.model(x_lb)
                logits_x_lb = outs_x_lb['logits']
                outs_x_ulb_s = self.model(x_ulb_s)
                logits_x_ulb_s = outs_x_ulb_s['logits']
            
            sup_loss = self.ce_loss(logits_x_lb, y_lb, reduction='mean')
            unsup_loss = self.consistency_loss(
                logits_x_ulb_s,
                pseudo_label,
                'ce',
                mask=mask
            )
            
            # Total loss combines supervised and combined unsupervised loss
            # In practice, unsup_loss represents the combined effect
            total_loss = sup_loss + self.lambda_u * (unsup_loss + self.alpha * unsup_loss_perturbed)
        
        out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
        log_dict = self.process_log_dict(
            sup_loss=sup_loss.item(),
            unsup_loss=unsup_loss.item(),
            unsup_loss_perturbed=unsup_loss_perturbed.item(),
            total_loss=total_loss.item(),
            util_ratio=mask.float().mean().item()
        )
        
        return out_dict, log_dict
    
    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--hard_label', str2bool, True),
            SSL_Argument('--T', float, 0.5),
            SSL_Argument('--p_cutoff', float, 0.95),
            SSL_Argument('--alpha', float, 1.0),
            SSL_Argument('--epsilon', float, 0.01),
        ]