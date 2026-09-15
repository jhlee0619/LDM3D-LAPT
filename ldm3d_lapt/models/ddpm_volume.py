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
from ldm3d_lapt.models.ddpm import DDPM, LatentDiffusion


__conditioning_keys__ = {'concat': 'c_concat',
                         'crossattn': 'c_crossattn',
                         'adm': 'y'}

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class LatentDiffusionVolume(LatentDiffusion):
    """main class"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        ckpt_path = kwargs.pop("ckpt_path", None)
        if ckpt_path is None:
            raise ValueError("ckpt_path cannot be None - must provide checkpoint path for initialization")
        
        self.temporal_length = kwargs.pop('unet_config', {}).pop('temporal_length', 32)

    @torch.no_grad()
    def get_input(self, batch, k, return_first_stage_outputs=False, force_c_encode=False,
                  cond_key=None, return_original_cond=False, bs=None):
        x = batch[k]
        x = x.to(memory_format=torch.contiguous_format).float()
        if bs is not None:
            x = x[:bs]
        x = rearrange(x, 'b c h w d -> (b d) c h w')
        x = x.to(self.device)
        encoder_posterior = self.encode_first_stage(x)
        z = self.get_first_stage_encoding(encoder_posterior).detach()

        if self.model.conditioning_key is not None:
            if cond_key is None:
                cond_key = self.cond_stage_key
            if cond_key != self.first_stage_key:
                if cond_key in ['caption', 'coordinates_bbox']:
                    xc = batch[cond_key]
                elif cond_key == 'class_label':
                    xc = batch
                else:
                    xc = batch[cond_key].to(self.device)
            else:
                xc = x
            if bs is not None:
                xc = xc[:bs]
            xc = rearrange(xc, 'b c h w d -> (b d) c h w')
            if not self.cond_stage_trainable or force_c_encode:
                if isinstance(xc, dict) or isinstance(xc, list):
                    # import pudb; pudb.set_trace()
                    c = self.get_learned_conditioning(xc)
                else:
                    c = self.get_learned_conditioning(xc.to(self.device))
            else:
                c = xc
            if self.use_positional_encodings:
                pos_x, pos_y = self.compute_latent_shifts(batch)
                ckey = __conditioning_keys__[self.model.conditioning_key]
                c = {ckey: c, 'pos_x': pos_x, 'pos_y': pos_y}

        else:
            c = None
            xc = None
            if self.use_positional_encodings:
                pos_x, pos_y = self.compute_latent_shifts(batch)
                c = {'pos_x': pos_x, 'pos_y': pos_y}
        mask = batch['mask']
        lesion = batch['lesion']
        if bs is not None:
            mask = mask[:bs]
            lesion = lesion[:bs]
        mask = rearrange(mask, 'b c h w d -> (b d) c h w')
        lesion = rearrange(lesion, 'b c h w d -> (b d) c h w')
        out = [z, c, mask, lesion]
        if return_first_stage_outputs:
            xrec = self.decode_first_stage(z)
            out.extend([x, xrec])
        if return_original_cond:
            out.append(xc)
        return out

    def shared_step(self, batch, **kwargs):
        x, c, mask, lesion = self.get_input(batch, self.first_stage_key)
        kwargs.update({'mask': mask, 'lesion': lesion})
        loss = self(x, c, **kwargs)
        return loss
    
    def forward(self, x, c, *args, **kwargs):
        t = torch.randint(0, self.num_timesteps, (x.shape[0]//self.temporal_length,), device=self.device).long()
        t = t.unsqueeze(-1).repeat(1, self.temporal_length)
        t = rearrange(t, 'b d -> (b d)')
        if self.model.conditioning_key is not None:
            assert c is not None
            if self.cond_stage_trainable:
                c = self.get_learned_conditioning(c)
            if self.shorten_cond_schedule:  # TODO: drop this option
                tc = self.cond_ids[t].to(self.device)
                c = self.q_sample(x_start=c, t=tc, noise=torch.randn_like(c.float()))
        return self.p_losses(x, c, t, *args, **kwargs)

    def p_losses(self, x_start, cond, t, noise=None, mask=None, lesion=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
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
            
        loss_dict.update({f'{prefix}/loss': loss})

        return loss, loss_dict

    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=4, sample=True, ddim_steps=200, ddim_eta=1., return_keys=None,
                   quantize_denoised=False, inpaint=False, plot_denoise_rows=False, plot_progressive_rows=False,
                   plot_diffusion_rows=False, **kwargs):
        def normalize_for_vis(img, 
                            dwi_mean=None, dwi_std=None, 
                            adc_mean=None, adc_std=None):
            # Denormalize z-score
            if img.shape[1] == 2:  # DWI+ADC case
                img_dwi = img[:, 0:1] * dwi_std[:, None, None, None, None] + dwi_mean[:, None, None, None, None]
                img_adc = img[:, 1:2] * adc_std[:, None, None, None, None] + adc_mean[:, None, None, None, None]

                B, C, H, W, D = img_dwi.shape
                q = torch.tensor([0.005, 0.995], dtype=img_dwi.dtype, device=img_dwi.device)
                flat_dwi = img_dwi.flatten(start_dim=2)  # (B, C, H*W*D)
                dwi_q = torch.quantile(flat_dwi, q, dim=2)  # (B, C, 2)
                dwi_min = dwi_q[0].view(B, C, 1, 1, 1)
                dwi_max = dwi_q[1].view(B, C, 1, 1, 1)
                flat_adc = img_adc.flatten(start_dim=2)
                adc_q = torch.quantile(flat_adc, q, dim=2)
                adc_min = adc_q[0].view(B, C, 1, 1, 1)
                adc_max = adc_q[1].view(B, C, 1, 1, 1)

                img_dwi = (img_dwi - dwi_min) / (dwi_max - dwi_min + 1e-8)
                img_adc = (img_adc - adc_min) / (adc_max - adc_min + 1e-8)
                img = torch.cat([img_dwi, img_adc], dim=1)
            return img

        N_ = N * self.temporal_length
        use_ddim = ddim_steps is not None

        log = dict()
        z, c, mask, lesion, x, xrec, xc = self.get_input(batch, self.first_stage_key,
                                           return_first_stage_outputs=True,
                                           force_c_encode=True,
                                           return_original_cond=True,
                                           bs=N)
        N = x.shape[0] // self.temporal_length
        N_ = N * self.temporal_length
        
        dwi_mean = batch.get('dwi_mean', None)
        dwi_std = batch.get('dwi_std', None)
        adc_mean = batch.get('adc_mean', None)
        adc_std = batch.get('adc_std', None)
        dwi_mean = dwi_mean[:N] if dwi_mean is not None else None
        dwi_std = dwi_std[:N] if dwi_std is not None else None
        adc_mean = adc_mean[:N] if adc_mean is not None else None
        adc_std = adc_std[:N] if adc_std is not None else None
        
        x = rearrange(x, '(b d) c h w -> b c h w d', d=self.temporal_length)
        xc = rearrange(xc, '(b d) c h w -> b c h w d', d=self.temporal_length)
        mask = rearrange(mask, '(b d) c h w -> b c h w d', d=self.temporal_length)
        lesion = rearrange(lesion, '(b d) c h w -> b c h w d', d=self.temporal_length)
        
        lesion_per_slice = lesion.sum(dim=(1, 2, 3))
        max_lesion_slice_idx = torch.argmax(lesion_per_slice, dim=1)
        half_window = self.temporal_length // 2
        window_size = 2 * half_window
        num_slices = lesion.shape[-1]
        start_indices = max_lesion_slice_idx - half_window
        start_indices = torch.clamp(start_indices, 0, num_slices - window_size)
        max_lesion_slice_ids = start_indices.unsqueeze(1) + torch.arange(window_size, device=start_indices.device).unsqueeze(0)
        
        inputs = normalize_for_vis(x[:N_], dwi_mean=dwi_mean, dwi_std=dwi_std,
                                          adc_mean=adc_mean, adc_std=adc_std)

        inputs = torch.stack([inputs[i, :, :, :, max_lesion_slice_ids[i]] for i in range(N)])
        xc = torch.stack([xc[i, :, :, :, max_lesion_slice_ids[i]] for i in range(N)])
        mask = torch.stack([mask[i, :, :, :, max_lesion_slice_ids[i]] for i in range(N)])
        inputs = rearrange(inputs, 'b c h w d -> (b d) c h w')
        xc = rearrange(xc, 'b c h w d -> (b d) c h w')
        mask = rearrange(mask, 'b c h w d -> (b d) c h w')
        
        log["conditioning"] = xc
        log["inputs"] = inputs

        if sample:
            # get denoise row
            with self.ema_scope("Plotting"):
                samples, z_denoise_row = self.sample_log(cond=c,batch_size=N_,ddim=use_ddim,
                                                         ddim_steps=ddim_steps,eta=ddim_eta)
            x_samples = self.decode_first_stage(samples)
            x_samples = rearrange(x_samples, '(b d) c h w -> b c h w d', d=self.temporal_length)
            x_samples = normalize_for_vis(x_samples, dwi_mean=dwi_mean, dwi_std=dwi_std,
                                               adc_mean=adc_mean, adc_std=adc_std)
            x_samples = torch.stack([x_samples[i, :, :, :, max_lesion_slice_ids[i]] for i in range(N)])
            x_samples = rearrange(x_samples, 'b c h w d -> (b d) c h w')
            log["samples"] = x_samples

        if return_keys:
            if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
                return log
            else:
                return {key: log[key] for key in return_keys}
        return log

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        if self.cond_stage_trainable:
            params = params + list(self.cond_stage_model.parameters())
        # params = []
        # for name, p in self.model.named_parameters():
        #     if 'tem' in name or 'temporal' in name:
        #         params.append(p)
        # if self.cond_stage_trainable:
        #     print(f"{self.__class__.__name__}: Also optimizing conditioner params!")
        #     cond_params = []
        #     for name, p in self.cond_stage_model.named_parameters():
        #         if 'tem' in name or 'temporal' in name:
        #             cond_params.append(p)
        #     params = params + cond_params
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
