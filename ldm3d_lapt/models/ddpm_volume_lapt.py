"""
wild mixture of
https://github.com/lucidrains/denoising-diffusion-pytorch/blob/7706bdfc6f527f58d33f84b7b522e61e6e3164b3/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
https://github.com/openai/improved-diffusion/blob/e94489283bb876ac1477d5dd7709bbbd2d9902ce/improved_diffusion/gaussian_diffusion.py
https://github.com/CompVis/taming-transformers
-- merci
"""
import re
import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR
from einops import rearrange, repeat
from contextlib import contextmanager
from functools import partial
from tqdm import tqdm
from torchvision.utils import make_grid
from pytorch_lightning.utilities.distributed import rank_zero_only

from ldm3d_lapt.util import log_txt_as_img, exists, default, ismap, isimage, mean_flat, count_params, instantiate_from_config
from ldm3d_lapt.modules.common import LitEma
from ldm3d_lapt.modules.common import normal_kl, DiagonalGaussianDistribution
from ldm3d_lapt.models.autoencoder import VQModelInterface, IdentityFirstStage, AutoencoderKL
from ldm3d_lapt.modules.util import make_beta_schedule, extract_into_tensor, noise_like
from ldm3d_lapt.models.ddim import DDIMSampler
from ldm3d_lapt.models.ddpm_volume import LatentDiffusionVolume


__conditioning_keys__ = {'concat': 'c_concat',
                         'crossattn': 'c_crossattn',
                         'adm': 'y'}

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class LatentDiffusionVolumeLAPT(LatentDiffusionVolume):
    """main class"""
    def __init__(self,
                 segmentation_model_config=None,
                 img_loss_weight=1.0,
                 seg_loss_weight=1.0,
                 lesion_loss_weight=10.0,
                 consecutive_slices: int = 8,
                 use_gradient_checkpointing: bool = True,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        ckpt_path = kwargs.pop("ckpt_path", None)
        if ckpt_path is None:
            raise ValueError("ckpt_path cannot be None - must provide checkpoint path for initialization")
        
        self.temporal_length = kwargs.pop('unet_config', {}).pop('temporal_length', 32)
        self.consecutive_slices = consecutive_slices

        self.segmentation_model = None
        if segmentation_model_config is not None:
            self.segmentation_model = instantiate_from_config(segmentation_model_config)

        self.img_loss_weight = img_loss_weight
        self.seg_loss_weight = seg_loss_weight
        self.lesion_loss_weight = lesion_loss_weight
        
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        if self.use_gradient_checkpointing:
            self._enable_gradient_checkpointing()

    def _enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency"""
        try:
            if hasattr(self.model, 'enable_gradient_checkpointing'):
                self.model.enable_gradient_checkpointing()
            elif hasattr(self.model, 'set_use_memory_efficient_attention_xformers'):
                self.model.set_use_memory_efficient_attention_xformers(True)
            print("Gradient checkpointing enabled for memory efficiency")
        except Exception as e:
            print(f"Could not enable gradient checkpointing: {e}")

    def register_schedule(self,
                          given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        super().register_schedule(given_betas, beta_schedule, timesteps, linear_start, linear_end, cosine_s)

        img_weights = torch.ones(self.num_timesteps, dtype=torch.float32)
        self.register_buffer('img_weights', img_weights, persistent=False)
        lesion_weights = torch.ones(self.num_timesteps, dtype=torch.float32)
        self.register_buffer('lesion_weights', lesion_weights, persistent=False)
        
        self.shorten_cond_schedule = self.num_timesteps_cond > 1
        if self.shorten_cond_schedule:
            self.make_cond_schedule()

    def p_losses(self, x_start, cond, t, noise=None, mask=None, lesion=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        
        # Use gradient checkpointing for model forward pass if enabled
        if self.use_gradient_checkpointing and self.training:
            model_output = torch.utils.checkpoint.checkpoint(
                self.apply_model, x_noisy, t, cond, use_reentrant=False
            )
        else:
            model_output = self.apply_model(x_noisy, t, cond)

        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        else:
            raise NotImplementedError()

        loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3])
        loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})

        logvar_t = self.logvar[t].to(self.device)
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        # loss = loss_simple / torch.exp(self.logvar) + self.logvar
        if self.learn_logvar:
            loss_dict.update({f'{prefix}/loss_gamma': loss.mean()})
            loss_dict.update({'logvar': self.logvar.data.mean()})
        
        loss = self.l_simple_weight * loss.mean()

        loss_vlb = self.get_loss(model_output, target, mean=False).mean(dim=(1, 2, 3))
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})
        loss += (self.original_elbo_weight * loss_vlb)
        
        # clean image (at 0)
        if self.parameterization == "eps":
            x_start_pred = self.predict_start_from_noise(x_noisy, t=t, noise=model_output)
        elif self.parameterization == "x0":
            x_start_pred = model_output
        else:
            raise NotImplementedError()
        
        x_start_pred, x_start, lesion, t = self._extract_consecutive_slices(x_start_pred, x_start, lesion, t)

        if self.use_gradient_checkpointing and self.training:
            img_pred = torch.utils.checkpoint.checkpoint(
                self.differentiable_decode_first_stage, x_start_pred, use_reentrant=False
            )
            img_target = torch.utils.checkpoint.checkpoint(
                self.differentiable_decode_first_stage, x_start, use_reentrant=False
            )
        else:
            img_pred = self.differentiable_decode_first_stage(x_start_pred)
            img_target = self.differentiable_decode_first_stage(x_start)
        
        # calculate mse loss between img_pred and target_img and weighted by mask and lesion
        loss_img = self.get_loss(img_pred, img_target, mean=False)
        
        loss_image = (loss_img).sum(dim=(1, 2, 3))
        # loss_image = (self.img_weights[t] * loss_image)
        loss_image = loss_image.sum()
        loss_image = loss_image / mask.sum()
        loss_dict.update({f'{prefix}/loss_image': loss_image})

        if self.img_loss_weight > 0:
            loss += (self.img_loss_weight * loss_image)
    
        loss_lesion = (loss_img * lesion).sum(dim=(1, 2, 3))
        lesion_sum_per_batch = lesion.sum(dim=(1, 2, 3))
        valid_idx = lesion_sum_per_batch > 0
        if valid_idx.any():
            loss_lesion_per_batch = (loss_img * lesion).sum(dim=(1, 2, 3))[valid_idx] / lesion_sum_per_batch[valid_idx]
            # loss_lesion_per_batch = (self.lesion_weights[t[valid_idx]] * loss_lesion_per_batch)
            loss_lesion = loss_lesion_per_batch.sum()
        else:
            loss_lesion = torch.tensor(0., device=loss_img.device)
        loss_dict.update({f'{prefix}/loss_lesion': loss_lesion})

        if self.lesion_loss_weight > 0:
            loss += (self.lesion_loss_weight * loss_lesion)

        # Process segmentation loss if segmentation model exists
        if self.segmentation_model is not None:
            img_pred_seg = rearrange(img_pred, '(b d) c h w -> b c h w d', d=self.consecutive_slices)
            lesion_seg = rearrange(lesion, '(b d) c h w -> b c h w d', d=self.consecutive_slices)
            
            # Use gradient checkpointing for segmentation model if enabled
            if self.use_gradient_checkpointing and self.training:
                loss_seg = torch.utils.checkpoint.checkpoint(
                    self.segmentation_model.calculate_loss, img_pred_seg, lesion_seg, alpha=0.1, beta=0.9, gamma_f=3.0, use_reentrant=False
                )
            else:
                loss_seg = self.segmentation_model.calculate_loss(img_pred_seg, lesion_seg, alpha=0.1, beta=0.9, gamma_f=3.0)
            
            loss_dict.update({f'{prefix}/loss_seg': loss_seg})
            
            if self.seg_loss_weight > 0:
                loss += (self.seg_loss_weight * loss_seg)
            
            # Clean up segmentation tensors
            del img_pred_seg, lesion_seg

        loss_dict.update({f'{prefix}/loss': loss})

        return loss, loss_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        if self.cond_stage_trainable:
            params = params + list(self.cond_stage_model.parameters())
        if self.learn_logvar:
            print('Diffusion model optimizing logvar')
            params.append(self.logvar)
        opt = torch.optim.AdamW(params, lr=lr)
        if self.use_scheduler:
            assert 'target' in self.scheduler_config
            scheduler = instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    'scheduler': LambdaLR(opt, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                }]
            return [opt], scheduler
        return opt

    def _extract_consecutive_slices(self, x_start_pred, x_start, lesion, t):
        """
        Extract consecutive slices from volume tensors for processing.
        
        Args:
            x_start_pred: Predicted start tensor
            x_start: Ground truth start tensor  
            lesion: Lesion mask tensor
            t: Time tensor
            
        Returns:
            Tuple of (x_start_pred, x_start, lesion, t) with consecutive slices extracted
        """
        # Rearrange to separate batch and depth dimensions
        x_start_pred = rearrange(x_start_pred, '(b d) c h w -> b c h w d', d=self.temporal_length)
        x_start = rearrange(x_start, '(b d) c h w -> b c h w d', d=self.temporal_length)
        lesion = rearrange(lesion, '(b d) c h w -> b c h w d', d=self.temporal_length)
        t = rearrange(t, '(b d) -> b d', d=self.temporal_length)
        
        # Get tensor dimensions
        b, c, h, w, d = x_start_pred.shape
        _, c_img, h_img, w_img, _ = lesion.shape
        max_start_idx = d - self.consecutive_slices  # Ensure we can get the desired consecutive slices
        
        if max_start_idx > 0:
            # Generate start indices based on lesion distribution
            start_indices = torch.zeros(b, dtype=torch.long, device=x_start_pred.device)
            
            for batch_idx in range(b):
                # Check which slices have lesions (sum over spatial dimensions)
                lesion_per_slice = lesion[batch_idx].sum(dim=(0, 1, 2))  # Sum over c, h, w dimensions
                lesion_slices = (lesion_per_slice > 0).nonzero(as_tuple=True)[0]  # Slices with lesions
                non_lesion_slices = (lesion_per_slice == 0).nonzero(as_tuple=True)[0]  # Slices without lesions
                
                # Filter slices to ensure we can get consecutive slices
                valid_lesion_slices = lesion_slices[lesion_slices <= max_start_idx]
                valid_non_lesion_slices = non_lesion_slices[non_lesion_slices <= max_start_idx]
                
                # Decide whether to sample from lesion slices (2/3) or non-lesion slices (1/3)
                # use_lesion_slice = torch.rand(1, device=x_start_pred.device).item() < (2/3)
                use_lesion_slice = torch.rand(1, device=x_start_pred.device).item() < 1
                
                if use_lesion_slice and len(valid_lesion_slices) > 0:
                    # Select from lesion slices
                    selected_idx = torch.randint(0, len(valid_lesion_slices), (1,), device=x_start_pred.device).item()
                    start_indices[batch_idx] = valid_lesion_slices[selected_idx]
                elif not use_lesion_slice and len(valid_non_lesion_slices) > 0:
                    # Select from non-lesion slices
                    selected_idx = torch.randint(0, len(valid_non_lesion_slices), (1,), device=x_start_pred.device).item()
                    start_indices[batch_idx] = valid_non_lesion_slices[selected_idx]
                else:
                    # Fallback: if no valid slices in the preferred category, use the other category
                    # or random selection if both categories are empty
                    if len(valid_lesion_slices) > 0:
                        selected_idx = torch.randint(0, len(valid_lesion_slices), (1,), device=x_start_pred.device).item()
                        start_indices[batch_idx] = valid_lesion_slices[selected_idx]
                    elif len(valid_non_lesion_slices) > 0:
                        selected_idx = torch.randint(0, len(valid_non_lesion_slices), (1,), device=x_start_pred.device).item()
                        start_indices[batch_idx] = valid_non_lesion_slices[selected_idx]
                    else:
                        # Complete fallback to random selection
                        start_indices[batch_idx] = torch.randint(0, max_start_idx + 1, (1,), device=x_start_pred.device).item()
            
            # Create tensors to store the selected slices for each batch
            slice_len = self.consecutive_slices
            x_start_pred_selected = torch.zeros(b, c, h, w, slice_len, device=x_start_pred.device)
            x_start_selected = torch.zeros(b, c, h, w, slice_len, device=x_start.device)
            lesion_selected = torch.zeros(b, c_img, h_img, w_img, slice_len, device=lesion.device)
            t_selected = torch.zeros(b, slice_len, device=t.device)
            
            # Extract consecutive slices for each batch
            for i in range(b):
                start_idx = start_indices[i]
                end_idx = start_idx + self.consecutive_slices
                x_start_pred_selected[i] = x_start_pred[i, :, :, :, start_idx:end_idx]
                x_start_selected[i] = x_start[i, :, :, :, start_idx:end_idx]
                lesion_selected[i] = lesion[i, :, :, :, start_idx:end_idx]
                t_selected[i] = t[i, start_idx:end_idx]
            
            x_start_pred = x_start_pred_selected
            x_start = x_start_selected
            t = t_selected
            lesion = lesion_selected
        else:
            # If d < consecutive_slices, pad with zeros or take all available slices
            slice_len = self.consecutive_slices
            x_start_pred = x_start_pred[:, :, :, :, :slice_len] if d >= slice_len else torch.nn.functional.pad(x_start_pred, (0, slice_len-d))
            x_start = x_start[:, :, :, :, :slice_len] if d >= slice_len else torch.nn.functional.pad(x_start, (0, slice_len-d))
            lesion = lesion[:, :, :, :, :slice_len] if d >= slice_len else torch.nn.functional.pad(lesion, (0, slice_len-d))
            t = t[:, :slice_len] if d >= slice_len else torch.nn.functional.pad(t, (0, slice_len-d))

        # Rearrange back to original format
        x_start_pred = rearrange(x_start_pred, 'b c h w d -> (b d) c h w')
        x_start = rearrange(x_start, 'b c h w d -> (b d) c h w')
        lesion = rearrange(lesion, 'b c h w d -> (b d) c h w')
        t = rearrange(t, 'b d -> (b d)')
        
        return x_start_pred, x_start, lesion, t
