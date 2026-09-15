import numpy as np
import torch
from monai import transforms
from monai.data import MetaTensor
from scipy.ndimage import morphology


class FinalSliceTransform(transforms.MapTransform):
    def __init__(
        self,
        keys = ('dwi', 'adc', 'ctp', 'mask'),
        rand_gaussian_noise: bool = False,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.rand_gaussian_noise = rand_gaussian_noise

    def __call__(self, data):
        d = dict(data)
        
        if 'dwi' in d:
            dwi = d['dwi']
            adc = d['adc']
            mask = d['mask']
            
            dwi_mean, dwi_std = d['dwi_mean'], d['dwi_std']
            adc_mean, adc_std = d['adc_mean'], d['adc_std']
            
            dwi = (dwi - dwi_mean) / (dwi_std + 1e-8)
            adc = (adc - adc_mean) / (adc_std + 1e-8)
            
            if self.rand_gaussian_noise:
                rand_apply = np.random.rand()
                if rand_apply < 0.5:
                    dwi_noise = np.random.normal(0, 0.2, dwi.shape)
                    adc_noise = np.random.normal(0, 0.2, adc.shape)
                    dwi = dwi + dwi_noise
                    adc = adc + adc_noise
            
            mri = np.concatenate([dwi, adc], axis=0)
            mri = mri * mask
            mri = np.clip(mri, -3, 3)
            
            d['mri'] = mri
            d['mask'] = mask
            d['dwi_mean'] = float(dwi_mean)
            d['dwi_std'] = float(dwi_std)
            d['adc_mean'] = float(adc_mean)
            d['adc_std'] = float(adc_std)
            
        if 'ct' in d:
            ct = d['ct']
            mask = d['mask']
            
            ct_minv, ct_maxv = 0., 200.
            
            ct = np.clip(ct, ct_minv, ct_maxv)
            ct = (ct - ct_minv) / (ct_maxv - ct_minv + 1e-8)
            ct = ct * mask
            ct = ct * 2 - 1
            
            d['ct'] = ct
            d['mask'] = mask
            d['ct_minv'] = float(ct_minv)
            d['ct_maxv'] = float(ct_maxv)
            
        if 'ctp' in d:
            ctp = d['ctp']
            mask = d['mask']
            
            ctp_minv, ctp_maxv = 0., 200.
            
            ctp = np.clip(ctp, ctp_minv, ctp_maxv)
            ctp = (ctp - ctp_minv) / (ctp_maxv - ctp_minv + 1e-8)
            ctp = ctp * mask
            ctp = ctp * 2 - 1
            
            d['ctp'] = ctp
            d['mask'] = mask
            d['ctp_minv'] = float(ctp_minv)
            d['ctp_maxv'] = float(ctp_maxv)
            
        d['lesion'] = d['lesion'].astype(np.float32)
        # lesion = np.zeros((2, *d['lesion'].shape[1:]), dtype=np.int64)
        # lesion[0, d['lesion'][0] == 0] = 1
        # lesion[1, d['lesion'][0] == 1] = 1
        # d['lesion'] = lesion
        
        return d

def get_train_transforms():
    train_transform = transforms.Compose(
        [
            transforms.EnsureChannelFirstd(keys=["dwi", "adc", "ct", "mask", "lesion"], channel_dim='no_channel', allow_missing_keys=True),
            transforms.RandFlipd(keys=["dwi", "adc", "ct", "ctp", "mask", "lesion"], prob=0.5, spatial_axis=0, allow_missing_keys=True),
            transforms.RandFlipd(keys=["dwi", "adc", "ct", "ctp", "mask", "lesion"], prob=0.5, spatial_axis=1, allow_missing_keys=True),
            transforms.RandAffined(
                keys=["dwi", "adc", "ct", "ctp", "mask", "lesion"],
                mode=("bilinear", "bilinear", "bilinear", "bilinear", "nearest", "nearest"),
                prob=0.9,
                rotate_range=(np.pi/16, np.pi/16),
                translate_range=(32, 32),
                scale_range=(0.001, 0.001),
                padding_mode="border",
                allow_missing_keys=True,
            ),
            FinalSliceTransform(),
            transforms.ToTensord(keys=["dwi", "adc", "ct", "mri", "ctp", "mask", "lesion", "gt", "input"], allow_missing_keys=True),
        ]
    )
    return train_transform

def get_val_transforms():
    val_transform = transforms.Compose(
        [
            transforms.EnsureChannelFirstd(keys=["dwi", "adc", "ct", "mask", "lesion"], channel_dim='no_channel', allow_missing_keys=True),
            FinalSliceTransform(),
            transforms.ToTensord(keys=["dwi", "adc", "ct", "mri", "ctp", "mask", "lesion", "gt", "input"], allow_missing_keys=True),
        ]
    )
    return val_transform




class HandleMissingLesiond(transforms.MapTransform):
    def __init__(self, keys, allow_missing_keys: bool = False) -> None:
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            if key not in d:
                d[key] = np.zeros_like(d['mask'], dtype=np.float32)
        return d

class FillHolesSliceWised(transforms.MapTransform):
    def __init__(self, keys, allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            if key in d:
                mask = d[key]
                
                is_tensor = isinstance(mask, torch.Tensor)
                if is_tensor:
                    mask_np = mask.cpu().detach().numpy()
                else:
                    mask_np = np.asarray(mask)

                filled_mask_np = mask_np.copy()
                # Assuming mask is (C, H, W, D), fill holes on each (H, W) slice
                for c in range(filled_mask_np.shape[0]):
                    for z in range(filled_mask_np.shape[3]):
                        slice_ = filled_mask_np[c, :, :, z]
                        filled_slice = morphology.binary_fill_holes(slice_.astype(bool))
                        filled_mask_np[c, :, :, z] = filled_slice.astype(filled_mask_np.dtype)
                
                if is_tensor:
                    new_tensor = torch.from_numpy(filled_mask_np)
                    if isinstance(mask, MetaTensor):
                        d[key] = MetaTensor(new_tensor, meta=mask.meta).to(mask.device)
                    else:
                        d[key] = new_tensor.to(mask.device)
                else:
                    d[key] = filled_mask_np
        return d

class FinalVolumeTransform(transforms.MapTransform):
    def __init__(
        self,
        keys = ('dwi', 'adc', 'ct', 'ctp', 'mask', 'lesion'),
        rand_gaussian_noise: bool = False,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.rand_gaussian_noise = rand_gaussian_noise

    def __call__(self, data):
        d = dict(data)
        
        if 'dwi' in d and 'adc' in d:
            dwi = d['dwi']
            adc = d['adc']
            mask = d['mask']

            dwi[dwi < 0] = 0
            adc[adc < 0] = 0

            dwi_signal = dwi[mask > 0]
            adc_signal = adc[mask > 0]
            
            dwi_mean, dwi_std = np.mean(dwi_signal), np.std(dwi_signal)
            adc_mean, adc_std = np.mean(adc_signal), np.std(adc_signal)
            
            dwi = (dwi - dwi_mean) / (dwi_std + 1e-8)
            adc = (adc - adc_mean) / (adc_std + 1e-8)
            
            if self.rand_gaussian_noise:
                rand_apply = np.random.rand()
                if rand_apply < 0.5:
                    dwi_noise = np.random.normal(0, 0.2, dwi.shape)
                    adc_noise = np.random.normal(0, 0.2, adc.shape)
                    dwi += dwi_noise
                    adc += adc_noise
            
            mri = np.concatenate([dwi, adc], axis=0)
            mri = mri * mask
            mri = np.clip(mri, -3, 3)
            
            d['mri'] = mri
            d['mask'] = mask
            d['dwi_mean'] = float(dwi_mean)
            d['dwi_std'] = float(dwi_std)
            d['adc_mean'] = float(adc_mean)
            d['adc_std'] = float(adc_std)

        if 'ct' in d:
            ct = d['ct']
            mask = d['mask']
            ct_minv, ct_maxv = 0., 200.
            ct = np.clip(ct, ct_minv, ct_maxv)
            ct = (ct - ct_minv) / (ct_maxv - ct_minv + 1e-8)
            ct = ct * mask
            ct = ct * 2 - 1
            d['ct'] = ct
            d['ct_minv'] = float(ct_minv)
            d['ct_maxv'] = float(ct_maxv)
            
        if 'ctp' in d:
            ctp = d['ctp']
            mask = d['mask']
            ctp_minv, ctp_maxv = 0., 200.
            ctp = np.clip(ctp, ctp_minv, ctp_maxv)
            ctp = (ctp - ctp_minv) / (ctp_maxv - ctp_minv + 1e-8)
            ctp = ctp * mask
            ctp = ctp * 2 - 1
            d['ctp'] = ctp
            d['ctp_minv'] = float(ctp_minv)
            d['ctp_maxv'] = float(ctp_maxv)
            
        d['lesion'] = d['lesion'].astype(np.float32)
        return d

# Volume size the 3D stages train on: 192 x 224 in-plane (the latent is [48, 56]
# at f4) by 32 slices, matching `temporal_length` in the conditioner configs.
spatial_size = (192, 224, 32)


def get_train_transforms_3d():
    train_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["dwi", "adc", "ct", "ctp", "mask", "lesion"], allow_missing_keys=True, image_only=True),
            HandleMissingLesiond(keys=['lesion']),
            transforms.Lambdad(keys=("dwi", "adc"), func=lambda x: np.maximum(x, 0), allow_missing_keys=True),
            transforms.Lambdad(keys=("ctp"), func=lambda x: x.permute(3, 0, 1, 2), allow_missing_keys=True),
            transforms.EnsureChannelFirstd(keys=["dwi", "adc", "ct", "mask", "lesion"], allow_missing_keys=True, channel_dim='no_channel'),
            FillHolesSliceWised(keys=['mask'], allow_missing_keys=True),
            transforms.RandFlipd(keys=["dwi", "adc", "ct", "ctp", "mask", "lesion"], prob=0.5, spatial_axis=0, allow_missing_keys=True),
            transforms.RandFlipd(keys=["dwi", "adc", "ct", "ctp", "mask", "lesion"], prob=0.5, spatial_axis=1, allow_missing_keys=True),
            transforms.RandFlipd(keys=["dwi", "adc", "ct", "ctp", "mask", "lesion"], prob=0.5, spatial_axis=2, allow_missing_keys=True),
            transforms.RandAffined(
                keys=["dwi", "adc", "ct", "ctp", "mask", "lesion"],
                mode=("bilinear", "bilinear", "bilinear", "bilinear", "nearest", "nearest"),
                prob=0.9,
                rotate_range=(np.pi/48, np.pi/48, np.pi/12),
                translate_range=(10, 10, 3),
                scale_range=(0.1, 0.1, 0.03),
                padding_mode="border",
                allow_missing_keys=True,
            ),
            transforms.SpatialPadd(keys=["dwi", "adc", "ct", "ctp", "mask", "lesion"], spatial_size=spatial_size, allow_missing_keys=True),
            FinalVolumeTransform(),
            transforms.RandSpatialCropd(keys=["mri", "dwi", "adc", "ct", "ctp", "mask", "lesion"], roi_size=spatial_size, allow_missing_keys=True),
            transforms.ToTensord(keys=["mri", "dwi", "adc", "ct", "ctp", "mask", "lesion"], allow_missing_keys=True),
        ]
    )
    return train_transform

def get_val_transforms_3d():
    val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["dwi", "adc", "ct", "ctp", "mask", "lesion"], allow_missing_keys=True, image_only=True),
            HandleMissingLesiond(keys=['lesion']),
            transforms.Lambdad(keys=("dwi", "adc"), func=lambda x: np.maximum(x, 0), allow_missing_keys=True),
            transforms.Lambdad(keys=("ctp"), func=lambda x: x.permute(3, 0, 1, 2), allow_missing_keys=True),
            transforms.EnsureChannelFirstd(keys=["dwi", "adc", "ct", "mask", "lesion"], allow_missing_keys=True, channel_dim='no_channel'),
            FillHolesSliceWised(keys=['mask'], allow_missing_keys=True),
            transforms.SpatialPadd(keys=["dwi", "adc", "ct", "ctp", "mask", "lesion"], spatial_size=spatial_size, allow_missing_keys=True),
            FinalVolumeTransform(),
            transforms.CenterSpatialCropd(keys=["mri", "dwi", "adc", "ct", "ctp", "mask", "lesion"], roi_size=spatial_size, allow_missing_keys=True),
            transforms.ToTensord(keys=["mri", "dwi", "adc", "ct", "ctp", "mask", "lesion"], allow_missing_keys=True),
        ]
    )
    return val_transform
    
seg_spatial_size = (192, 224, 8)

def get_train_bigaug_transforms_3d():
    train_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["dwi", "adc", "ct", "ctp", "mask", "lesion"], allow_missing_keys=True, image_only=True),
            HandleMissingLesiond(keys=['lesion']),
            transforms.Lambdad(keys=("dwi", "adc"), func=lambda x: np.maximum(x, 0), allow_missing_keys=True),
            transforms.Lambdad(keys=("ctp"), func=lambda x: x.permute(3, 0, 1, 2), allow_missing_keys=True),
            transforms.EnsureChannelFirstd(keys=["dwi", "adc", "ct", "mask", "lesion"], allow_missing_keys=True, channel_dim='no_channel'),
            FillHolesSliceWised(keys=['mask'], allow_missing_keys=True),
            transforms.RandFlipd(keys=["dwi", "adc", "ct", "ctp", "mask", "lesion"], prob=0.5, spatial_axis=0, allow_missing_keys=True),
            transforms.RandFlipd(keys=["dwi", "adc", "ct", "ctp", "mask", "lesion"], prob=0.5, spatial_axis=1, allow_missing_keys=True),
            transforms.RandFlipd(keys=["dwi", "adc", "ct", "ctp", "mask", "lesion"], prob=0.5, spatial_axis=2, allow_missing_keys=True),
            transforms.RandGaussianSmoothd(
                keys=["dwi", "adc", "ct", "ctp"],
                sigma_x=(0.25, 1.0), sigma_y=(0.25, 1.0), sigma_z=(0.0625, 0.25),
                prob=0.3, allow_missing_keys=True,
            ),
            transforms.RandGaussianSharpend(
                keys=["dwi", "adc", "ct", "ctp"],
                sigma1_x=(0.25, 0.5), sigma1_y=(0.25, 0.5), sigma1_z=(0.0625, 0.125),
                sigma2_x=(0.8, 1.2), sigma2_y=(0.8, 1.2), sigma2_z=(0.2, 0.3),
                prob=0.3, allow_missing_keys=True,
            ),
            transforms.RandAdjustContrastd(
                keys=["dwi", "adc", "ct", "ctp"], gamma=(0.8, 2.0),
                prob=0.3, allow_missing_keys=True,
            ),
            transforms.RandShiftIntensityd(
                keys=["dwi", "adc", "ct", "ctp"], offsets=(-0.03, 0.03),
                prob=0.3, allow_missing_keys=True,
            ),
            transforms.RandScaleIntensityd(
                keys=["dwi", "adc", "ct", "ctp"], factors=(0.95, 1.05),
                prob=0.3, allow_missing_keys=True,
            ),
            transforms.RandAffined(
                keys=["dwi", "adc", "ct", "ctp", "mask", "lesion"],
                mode=("bilinear", "bilinear", "bilinear", "bilinear", "nearest", "nearest"),
                prob=0.5,
                rotate_range=(np.pi/48, np.pi/48, np.pi/12),
                translate_range=(10, 10, 3),
                scale_range=(0.2, 0.2, 0.05),
                padding_mode="border",
                allow_missing_keys=True,
            ),
            transforms.RandGridDistortiond(
                keys=["dwi", "adc", "ct", "ctp", "mask", "lesion"],
                distort_limit=(-0.01, 0.01),
                mode=("bilinear", "bilinear", "bilinear", "bilinear", "nearest", "nearest"),
                prob=0.3,
                allow_missing_keys=True,
            ),
            transforms.SpatialPadd(keys=["dwi", "adc", "ct", "ctp", "mask", "lesion"], spatial_size=spatial_size, allow_missing_keys=True),
            FinalVolumeTransform(rand_gaussian_noise=True),
            transforms.RandCropByPosNegLabeld(
                keys=["mri", "dwi", "adc", "ct", "ctp", "mask", "lesion"],
                label_key="lesion",
                spatial_size=seg_spatial_size,
                pos=2.0,
                neg=1.0,
                num_samples=1,
                allow_missing_keys=True,
            ),
            transforms.ToTensord(keys=["mri", "dwi", "adc", "ct", "ctp", "mask", "lesion"], allow_missing_keys=True),
        ]
    )
    return train_transform


def get_val_bigaug_transforms_3d():
    val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["dwi", "adc", "ct", "ctp", "mask", "lesion"], allow_missing_keys=True, image_only=True),
            HandleMissingLesiond(keys=['lesion']),
            transforms.Lambdad(keys=("dwi", "adc"), func=lambda x: np.maximum(x, 0), allow_missing_keys=True),
            transforms.Lambdad(keys=("ctp"), func=lambda x: x.permute(3, 0, 1, 2), allow_missing_keys=True),
            transforms.EnsureChannelFirstd(keys=["dwi", "adc", "ct", "mask", "lesion"], allow_missing_keys=True, channel_dim='no_channel'),
            FillHolesSliceWised(keys=['mask'], allow_missing_keys=True),
            FinalVolumeTransform(),
            transforms.RandCropByPosNegLabeld(
                keys=["mri", "dwi", "adc", "ct", "ctp", "mask", "lesion"],
                label_key="lesion",
                spatial_size=seg_spatial_size,
                pos=1.0,
                neg=0.0,
                num_samples=1,
                allow_missing_keys=True,
            ),
            transforms.ToTensord(keys=["mri", "dwi", "adc", "ct", "ctp", "mask", "lesion"], allow_missing_keys=True),
        ]
    )
    return val_transform
