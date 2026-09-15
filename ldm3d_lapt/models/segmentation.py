import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
from torchmetrics import Dice, JaccardIndex
from monai.losses import DiceLoss
from einops import rearrange, repeat

from ldm3d_lapt.util import instantiate_from_config

from ldm3d_lapt.modules.backbone_3d import Model3D as Model


class SegmentationModel3D(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 num_classes,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="mri",
                 colorize_nlabels=None,
                 monitor=None,
                 ):
        super().__init__()
        self.image_key = image_key
        self.model = Model(**ddconfig)
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.image_key = image_key
        if colorize_nlabels is not None:
            assert type(colorize_nlabels) == int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        # These metrics from torchmetrics are channel-first and work for 2D or 3D
        self.dice = Dice()  # binary segmentation needs no num_classes
        self.miou = JaccardIndex(task='binary')  # configured for binary segmentation

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def forward(self, input):
        pred = self.model(input)
        return pred

    def get_input(self, batch, k):
        x = batch[k]
        return x.float()

    def training_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        lesion_mask = self.get_input(batch, "lesion").long()  # cast to integer
        pred = self(x)
        pred = torch.sigmoid(pred)  # apply sigmoid

        # combined loss
        loss = combined_loss(
            y_true=lesion_mask,
            y_pred=pred,
            alpha=0.7,
            beta=0.3,
            gamma_f=2.0,
            smooth=1e-6,
            lambda_tversky=0.5,
            lambda_focal=0.5
        )

        # metrics, after thresholding
        threshold = 0.5
        pred_binary = (pred > threshold).float()
        dice_score = self.dice(pred_binary, lesion_mask)
        miou_score = self.miou(pred_binary, lesion_mask)

        # Logging
        self.log("train/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("train/dice", dice_score, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("train/miou", miou_score, prog_bar=True, logger=True, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        lesion_mask = self.get_input(batch, "lesion").long()  # cast to integer
        pred = self(x)
        pred = torch.sigmoid(pred)  # apply sigmoid

        # combined loss
        loss = combined_loss(
            y_true=lesion_mask,
            y_pred=pred,
            alpha=0.7,
            beta=0.3,
            gamma_f=2.0,
            smooth=1e-6,
            lambda_tversky=0.5,
            lambda_focal=0.5
        )

        # metrics, after thresholding
        threshold = 0.5
        pred_binary = (pred > threshold).float()
        dice_score = self.dice(pred_binary, lesion_mask)
        miou_score = self.miou(pred_binary, lesion_mask)

        # Logging
        self.log("val/loss", loss, prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/dice", dice_score, prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/miou", miou_score, prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)

        return {"val_loss": loss, "val_dice": dice_score, "val_miou": miou_score}

    def configure_optimizers(self):
        lr = self.learning_rate
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, betas=(0.5, 0.9))
        return optimizer

    def log_images(self, batch, **kwargs):
        def normalize_for_vis(img,
                            dwi_mean=None, dwi_std=None,
                            adc_mean=None, adc_std=None):
            # Denormalize z-score
            if img.shape[1] == 2:  # DWI+ADC case
                img_dwi = img[:, 0:1] * dwi_std[:, None, None, None, None] + dwi_mean[:, None, None, None, None]
                img_adc = img[:, 1:2] * adc_std[:, None, None, None, None] + adc_mean[:, None, None, None, None]

                B, C, H, W, D = img_dwi.shape
                q = torch.tensor([0.005, 0.995], dtype=img_dwi.dtype, device=img_dwi.device)
                flat_dwi = img_dwi.flatten(start_dim=2)  # (B, C, H*W)
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

        log = dict()
        x = self.get_input(batch, self.image_key)
        lesion = self.get_input(batch, "lesion").long()
        x = x.to(self.device)

        B = x.shape[0]
        dwi_mean = batch.get('dwi_mean', None)
        dwi_std = batch.get('dwi_std', None)
        adc_mean = batch.get('adc_mean', None)
        adc_std = batch.get('adc_std', None)

        lesion_per_slice = lesion.sum(dim=(1, 2, 3))
        max_lesion_slice_idx = torch.argmax(lesion_per_slice, dim=1)
        half_window = 4
        window_size = 2 * half_window
        num_slices = lesion.shape[-1]
        start_indices = max_lesion_slice_idx - half_window
        start_indices = torch.clamp(start_indices, 0, num_slices - window_size)
        max_lesion_slice_ids = start_indices.unsqueeze(1) + torch.arange(window_size, device=start_indices.device).unsqueeze(0)

        inputs = normalize_for_vis(x, dwi_mean=dwi_mean, dwi_std=dwi_std, adc_mean=adc_mean, adc_std=adc_std)

        pred = self(x)
        pred = torch.sigmoid(pred)  # Apply sigmoid for prediction

        inputs = torch.stack([inputs[i, :, :, :, max_lesion_slice_ids[i]] for i in range(B)])
        pred = torch.stack([pred[i, :, :, :, max_lesion_slice_ids[i]] for i in range(B)])
        lesion = torch.stack([lesion[i, :, :, :, max_lesion_slice_ids[i]] for i in range(B)])
        inputs = rearrange(inputs, 'b c h w d -> (b d) c h w')
        pred = rearrange(pred, 'b c h w d -> (b d) c h w')
        lesion = rearrange(lesion, 'b c h w d -> (b d) c h w')

        log["inputs"] = inputs
        log["predictions"] = pred
        log["lesion_masks"] = lesion.float()
        return log

    def predict(self, x):
        x = x.to(self.device)
        pred = self(x)
        return pred

class SegmentationGuide3D(SegmentationModel3D):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def calculate_loss(self, img_pred, img_target, alpha=0.7, beta=0.3, gamma_f=2.0, smooth=1e-6, lambda_tversky=0.5, lambda_focal=0.5):
        # Tversky / focal-Dice expect probabilities, so squash the logits the same
        # way training_step and validation_step do.
        mask_pred = torch.sigmoid(self.model(img_pred))

        loss = combined_loss(
            y_true=img_target,
            y_pred=mask_pred,
            alpha=alpha,
            beta=beta,
            gamma_f=gamma_f,
            smooth=smooth,
            lambda_tversky=lambda_tversky,
            lambda_focal=lambda_focal
        )
        return loss


def tversky_loss(y_true, y_pred, alpha=0.7, beta=0.3, smooth=1e-6):
    """
    Tversky loss.
    alpha: weight on false positives
    beta: weight on false negatives
    smooth: small constant for numerical stability
    """
    # true positives, false positives, false negatives
    TP = torch.sum(y_true * y_pred)
    FP = torch.sum((1 - y_true) * y_pred)
    FN = torch.sum(y_true * (1 - y_pred))

    # Tversky index
    tversky_index = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)

    # the loss is 1 - Tversky index
    return 1 - tversky_index

def focal_dice_loss(y_true, y_pred, gamma_f=2.0, smooth=1e-6):
    """
    Focal Dice loss.
    gamma_f: focal parameter
    smooth: small constant for numerical stability
    """
    # Dice coefficient
    intersection = torch.sum(y_true * y_pred)
    dice_coeff = (2. * intersection + smooth) / (torch.sum(y_true) + torch.sum(y_pred) + smooth)

    # Focal Dice loss.
    focal_dice = torch.pow((1. - dice_coeff), gamma_f) * (1. - dice_coeff)

    return focal_dice

def combined_loss(y_true, y_pred, alpha=0.7, beta=0.3, gamma_f=2.0, smooth=1e-6, lambda_tversky=0.5, lambda_focal=0.5):
    """
    Weighted sum of the Tversky and focal Dice losses.
    alpha: weight on false positives, in the Tversky term
    beta: weight on false negatives, in the Tversky term
    gamma_f: focal parameter of the focal Dice term
    lambda_tversky: weight of the Tversky term
    lambda_focal: weight of the focal Dice term
    """
    # Tversky loss.
    t_loss = tversky_loss(y_true, y_pred, alpha, beta, smooth)

    # Focal Dice loss.
    fdl_loss = focal_dice_loss(y_true, y_pred, gamma_f, smooth)

    # weighted sum of the two
    total_loss = lambda_tversky * t_loss + lambda_focal * fdl_loss
    return total_loss
