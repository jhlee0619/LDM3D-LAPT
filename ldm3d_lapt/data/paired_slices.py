import os
from glob import glob
import numpy as np
import json
import random
import ants
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torch.multiprocessing import Pool
from scipy.ndimage import morphology
from .transforms import *


# Optional second location searched when a file is missing under `dataroot`
# (e.g. a slower archive mount). Leave unset to disable the fallback.
dataroot_server = os.environ.get('DATAROOT_FALLBACK', '')

class PairedSliceDataset(Dataset):
    def __init__(self, dataroot, phase, transform, lesion=False, threshold=0, slice_idx=None):
        super().__init__()
        self.dataroot = dataroot
        self.phase = phase
        self.transform = transform
        self.lesion = lesion
        self.threshold = threshold

        self.dataset = []
        self.dataset_slice_idx_dict = {}
        
        self._collect_samples()

        if slice_idx is not None:
            self.dataset = self.dataset[:slice_idx]

        print(f'Total {phase} dataset size - volume: {len(self.dataset_slice_idx_dict)}, slice: {len(self.dataset)}')

    def __getitem__(self, index):
        dataroot, patient_id, study_date, series_id, i = self.dataset[index]
        ctp_path = os.path.join(dataroot, 'npy', patient_id, study_date, series_id, f'ctp_{i}.npy')
        dwi_path = os.path.join(dataroot, 'npy', patient_id, study_date, series_id, f'dwi_{i}.npy')
        adc_path = os.path.join(dataroot, 'npy', patient_id, study_date, series_id, f'adc_{i}.npy')
        mask_path = os.path.join(dataroot, 'npy', patient_id, study_date, series_id, f'mask_{i}.npy')
        lesion_path = os.path.join(dataroot, 'npy', patient_id, study_date, series_id, f'lesion_{i}.npy')
        dwi_stats_path = os.path.join(dataroot, 'npy', patient_id, study_date, series_id, f'dwi_stats.npy')
        adc_stats_path = os.path.join(dataroot, 'npy', patient_id, study_date, series_id, f'adc_stats.npy')
        
        if dataroot_server and (not os.path.exists(ctp_path) or not os.path.exists(dwi_path) or not os.path.exists(adc_path) or not os.path.exists(mask_path) or not os.path.exists(lesion_path)):
            ctp, dwi, adc, mask, lesion, dwi_mean, dwi_std, adc_mean, adc_std, patient_id, study_date, series_id, patient_dir, i = self._single_scan(dataroot, patient_id, study_date, series_id, i)
        else:
            try:
                patient_dir = os.path.join(dataroot, 'nifti', patient_id, study_date, series_id)
                ctp = np.load(ctp_path)
                dwi = np.load(dwi_path)
                adc = np.load(adc_path)
                mask = np.load(mask_path)
                lesion = np.load(lesion_path)
                dwi_mean, dwi_std = np.load(dwi_stats_path)
                adc_mean, adc_std = np.load(adc_stats_path)
            except:
                ctp, dwi, adc, mask, lesion, dwi_mean, dwi_std, adc_mean, adc_std, patient_id, study_date, series_id, patient_dir, i = self._single_scan(dataroot, patient_id, study_date, series_id, i)
            
        data = self.transform({'ctp': ctp, 'dwi': dwi, 'adc': adc, 'mask': mask, 'lesion': lesion, 'dwi_mean': dwi_mean, 'dwi_std': dwi_std, 'adc_mean': adc_mean, 'adc_std': adc_std})
        # data.update({'patient_id': patient_id, 'study_date': study_date, 'series_id': series_id, 'patient_dir': patient_dir, 'slice': i})
        data.update({'patient_id': patient_id, 'study_date': study_date, 'series_id': series_id, 'patient_dir': patient_dir, 'slice': i, 'center': dataroot.split('/')[-3]})
        
        return data

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.dataset)

    def _single_scan(self, dataroot, patient_id, study_date, series_id, i):
        patient_dir = os.path.join(dataroot, 'nifti', patient_id, study_date, series_id)
        os.makedirs(f'{dataroot}/npy/{patient_id}/{study_date}/{series_id}', exist_ok=True)
        ctp_path = os.path.join(patient_dir, 'ctp.nii.gz')
        dwi_path = os.path.join(patient_dir, 'dwi.nii.gz')
        adc_path = os.path.join(patient_dir, 'adc.nii.gz')
        mask_path = os.path.join(patient_dir, 'ctp_mask.nii.gz')
        lesion_path = os.path.join(patient_dir, 'lesion.nii.gz')

        if dataroot_server and (not os.path.exists(ctp_path) or not os.path.exists(dwi_path) or not os.path.exists(adc_path) or not os.path.exists(mask_path) or not os.path.exists(lesion_path)):
            ctp_path = ctp_path.replace(self.dataroot, dataroot_server)
            dwi_path = dwi_path.replace(self.dataroot, dataroot_server)
            adc_path = adc_path.replace(self.dataroot, dataroot_server)
            mask_path = mask_path.replace(self.dataroot, dataroot_server)
            lesion_path = lesion_path.replace(self.dataroot, dataroot_server)
        
        ctp = ants.image_read(ctp_path).numpy()
        dwi = ants.image_read(dwi_path).numpy()
        adc = ants.image_read(adc_path).numpy()
        mask = ants.image_read(mask_path).numpy().astype(np.uint8)
        lesion = ants.image_read(lesion_path).numpy() if os.path.exists(lesion_path) else np.zeros_like(mask)
        
        dwi[dwi < 0] = 0
        adc[adc < 0] = 0
        
        dwi_signal = dwi[mask > 0]
        adc_signal = adc[mask > 0]
        
        dwi_mean, dwi_std = np.mean(dwi_signal), np.std(dwi_signal)
        adc_mean, adc_std = np.mean(adc_signal), np.std(adc_signal)
        np.save(os.path.join(dataroot, 'npy', patient_id, study_date, series_id, f'dwi_stats.npy'), np.array([dwi_mean, dwi_std]))
        np.save(os.path.join(dataroot, 'npy', patient_id, study_date, series_id, f'adc_stats.npy'), np.array([adc_mean, adc_std]))
        
        dataset_slice_idx = self.dataset_slice_idx_dict[f'{patient_id}_{study_date}_{series_id}']['slice_idx']
        for slice_idx in dataset_slice_idx:
            ctp_slice = ctp[:, :, slice_idx]
            ctp_slice = ctp_slice.transpose(2, 0, 1)
            dwi_slice = dwi[:, :, slice_idx]
            adc_slice = adc[:, :, slice_idx]
            mask_slice = mask[:, :, slice_idx]
            mask_slice = morphology.binary_fill_holes(mask_slice)
            lesion_slice = lesion[:, :, slice_idx]
            
            np.save(os.path.join(dataroot, 'npy', patient_id, study_date, series_id, f'ctp_{slice_idx}.npy'), ctp_slice)
            np.save(os.path.join(dataroot, 'npy', patient_id, study_date, series_id, f'dwi_{slice_idx}.npy'), dwi_slice)
            np.save(os.path.join(dataroot, 'npy', patient_id, study_date, series_id, f'adc_{slice_idx}.npy'), adc_slice)
            np.save(os.path.join(dataroot, 'npy', patient_id, study_date, series_id, f'mask_{slice_idx}.npy'), mask_slice)
            np.save(os.path.join(dataroot, 'npy', patient_id, study_date, series_id, f'lesion_{slice_idx}.npy'), lesion_slice)
        
        ctp_slice = ctp[:, :, i]
        ctp_slice = ctp_slice.transpose(2, 0, 1)
        dwi_slice = dwi[:, :, i]
        adc_slice = adc[:, :, i]
        mask_slice = mask[:, :, i]
        mask_slice = morphology.binary_fill_holes(mask_slice)
        lesion_slice = lesion[:, :, i]
        
        return [ctp_slice, dwi_slice, adc_slice, mask_slice, lesion_slice, dwi_mean, dwi_std, adc_mean, adc_std, patient_id, study_date, series_id, patient_dir, i]
        
    def _collect_samples(self):
        root = self.dataroot
        
        dataset_slice_idx_dict_path = os.path.join(self.dataroot, 'splits', 'slice_index.json')
        json_path = os.path.join(self.dataroot, 'splits', f'{self.phase}_list.json')
        self.patient_list = json.load(open(json_path, 'r'))
        # self.patient_list = self.patient_list[:1]
        
        dataset_slice_idx_dict = json.load(open(dataset_slice_idx_dict_path, 'r'))
        dataset_slice_idx_dict = {k: v for k, v in dataset_slice_idx_dict.items() if k.split('_')[0] in self.patient_list}
        if self.lesion and self.phase != 'test':
            samples = [[root, *k.split('_'), slice_idx] for k, v in dataset_slice_idx_dict.items() for idx, slice_idx in enumerate(v['slice_idx_intersection']) if v['slice_n_intersection'][idx] >= self.threshold]
        else:
            samples = [[root, *k.split('_'), slice_idx] for k, v in dataset_slice_idx_dict.items() for slice_idx in v['slice_idx']]
        self.dataset += samples
        self.dataset_slice_idx_dict.update(dataset_slice_idx_dict)
        
        print(f'{self.phase} dataset size - patient: {len(self.patient_list)}, volume: {len(dataset_slice_idx_dict)}, slice: {len(samples)}')

    def drop_exist_patient(self, outdir):
        prev_len = len(self.dataset)
        dataset = []
        for data in self.dataset:
            dataroot, patient_id, study_date, series_id, i = data
            center = dataroot.split('/')[-3]
            patient_result_path = os.path.join(outdir, f'{center}_{patient_id}_{study_date}_{series_id}')
            if not os.path.exists(os.path.join(patient_result_path, f'adc_sample_0.nii.gz')):
                dataset.append(data)
        self.dataset = dataset
        print(f'Dropped {prev_len - len(self.dataset)} datasets')
        

class PairedSliceTrain(PairedSliceDataset):
    def __init__(self, dataroot, slice_idx=None):
        transform = get_train_transforms()
        super().__init__(dataroot, phase='train', transform=transform, slice_idx=slice_idx)




class PairedSliceTest(PairedSliceDataset):
    def __init__(self, dataroot, slice_idx=None):
        transform = get_val_transforms()
        super().__init__(dataroot, phase='test', transform=transform, slice_idx=slice_idx)
