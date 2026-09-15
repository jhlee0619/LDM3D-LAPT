import os
import glob
import numpy as np
import json
import random
import ants
import torch
from torch.utils.data import Dataset
from .transforms import get_train_transforms_3d, get_val_transforms_3d, get_train_bigaug_transforms_3d

# Optional second location searched when a file is missing under `dataroot`
# (e.g. a slower archive mount). Leave unset to disable the fallback.
dataroot_server = os.environ.get('DATAROOT_FALLBACK', '')

class PairedVolumeDataset(Dataset):
    def __init__(self, dataroot, phase, transform, slice_idx=None):
        super().__init__()
        self.dataroot = dataroot
        self.phase = phase
        self.transform = transform

        root = self.dataroot
        json_path = os.path.join(self.dataroot, 'splits', f'{self.phase}_list.json')
        self.patient_ids = json.load(open(json_path, 'r'))
        self.dataset = glob.glob(os.path.join(root, 'nifti', '*', '*', '*'))
        self.dataset = [p for p in self.dataset if p.split('/')[-3] in self.patient_ids]
        
        if slice_idx is not None:
            self.dataset = self.dataset[:slice_idx]
            
        print(f'Total {phase} dataset size - volume: {len(self.dataset)} / patient: {len(self.patient_ids)}')

    def __getitem__(self, index):
        patient_dir = self.dataset[index]
        patient_id, study_date, series_id = patient_dir.split('/')[-3:]
        
        ctp_path = os.path.join(patient_dir, 'ctp.nii.gz')
        dwi_path = os.path.join(patient_dir, 'dwi.nii.gz')
        adc_path = os.path.join(patient_dir, 'adc.nii.gz')
        mask_path = os.path.join(patient_dir, 'ctp_mask.nii.gz')
        lesion_path = os.path.join(patient_dir, 'lesion.nii.gz')

        # If files don't exist, try reading from the server path
        if dataroot_server and (not os.path.exists(ctp_path) or not os.path.exists(dwi_path) or not os.path.exists(adc_path) or not os.path.exists(mask_path)):
            ctp_path = ctp_path.replace(self.dataroot, dataroot_server)
            dwi_path = dwi_path.replace(self.dataroot, dataroot_server)
            adc_path = adc_path.replace(self.dataroot, dataroot_server)
            mask_path = mask_path.replace(self.dataroot, dataroot_server)
        if dataroot_server and (not os.path.exists(lesion_path)):
            lesion_path = lesion_path.replace(self.dataroot, dataroot_server)

        data = {
            'ctp': ctp_path,
            'dwi': dwi_path,
            'adc': adc_path,
            'mask': mask_path,
        }
        if os.path.exists(lesion_path):
            data['lesion'] = lesion_path
        data = self.transform(data)
        if isinstance(data, list):
            data = data[0]
            
        data.update({'patient_id': patient_id, 'study_date': study_date, 'series_id': series_id})
        
        return data

    def __len__(self):
        return len(self.dataset)

class PairedVolumeTrain(PairedVolumeDataset):
    def __init__(self, dataroot, use_big_aug=False, slice_idx=None):
        transform = get_train_bigaug_transforms_3d() if use_big_aug else get_train_transforms_3d()
        super().__init__(dataroot, phase='train', transform=transform, slice_idx=slice_idx)


class PairedVolumeTest(PairedVolumeDataset):
    def __init__(self, dataroot, slice_idx=None):
        transform = get_val_transforms_3d()
        super().__init__(dataroot, phase='test', transform=transform, slice_idx=slice_idx)
