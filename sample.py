import argparse, os, sys, glob
import copy
import hashlib
import random
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
import ants
# add path 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ldm3d_lapt.util import instantiate_from_config
from ldm3d_lapt.data.paired_volumes import PairedVolumeDataset
from ldm3d_lapt.data.transforms import get_val_transforms_3d
from ldm3d_lapt.models.ddim import DDIMSampler
from einops import rearrange

class PairedVolumeSample(PairedVolumeDataset):
    def __init__(self, dataroot, phase, transform, outdir_name, steps, start_idx=0, end_idx=None):
        self.dataroot = dataroot
        self.transform = transform
        
        all_dataset = sorted(glob.glob(f'{self.dataroot}/*/*/*'))
        self.dataset = []
        print(f'Found {len(all_dataset)} samples')
        for p in tqdm(all_dataset):
            patient_result_path = os.path.join(p, outdir_name, f'num_steps_{steps}')
            dwi_sample_path = os.path.join(patient_result_path, f'dwi_sample.nii.gz')
            adc_sample_path = os.path.join(patient_result_path, f'adc_sample.nii.gz')
            
            if not (os.path.exists(dwi_sample_path) and os.path.exists(adc_sample_path)):
                self.dataset.append(p)
        print(f'Filtered to {len(self.dataset)} samples')
        
        # Apply slicing if indices are provided
        if end_idx is None:
            end_idx = len(self.dataset)
        self.dataset = self.dataset[start_idx:end_idx]
        print(f'After slicing (start_idx={start_idx}, end_idx={end_idx}): {len(self.dataset)} samples')
        
        self.patient_list = [p.split('/')[-3] for p in self.dataset]
        
    def __getitem__(self, index):
        data = super().__getitem__(index)
        return data

def normalize_data(img, dwi_mean=None, dwi_std=None, adc_mean=None, adc_std=None):
    img_dwi = img[:, 0:1] * dwi_std[:, None, None, None] + dwi_mean[:, None, None, None]
    img_adc = img[:, 1:2] * adc_std[:, None, None, None] + adc_mean[:, None, None, None]
    img = torch.cat([img_dwi, img_adc], dim=1)
    return img


def stable_case_seed(base_seed, model_id, patient_id, study_date, series_id):
    payload = f"{base_seed}|{model_id}|{patient_id}__{study_date}__{series_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31 - 1)


def set_inference_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def save_result(dataroot, outdir_name, steps, patient_id, study_date, series_id, sample):
    patient_dir = os.path.join(dataroot, patient_id, study_date, series_id)
    patient_result_path = os.path.join(
        patient_dir, outdir_name, f"num_steps_{steps}"
    )
    os.makedirs(patient_result_path, exist_ok=True)
    template = ants.image_read(os.path.join(patient_dir, "gt", "dwi.nii.gz"))
    template.new_image_like(sample[0]).to_file(
        os.path.join(patient_result_path, "dwi_sample.nii.gz")
    )
    template.new_image_like(sample[1]).to_file(
        os.path.join(patient_result_path, "adc_sample.nii.gz")
    )
        
def load_model_from_config(config, sd):
    model = instantiate_from_config(config)
    print(f'model restored for inference')
    model.load_state_dict(sd,strict=False)
    model.cuda()
    model.eval()
    return model

def load_model(config, ckpt, gpu, eval_mode):
    if ckpt:
        print(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt, map_location="cpu")
        global_step = pl_sd["global_step"]
    else:
        pl_sd = {"state_dict": None}
        global_step = None
    model = load_model_from_config(config.model,
                                   pl_sd["state_dict"])

    return model, global_step

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        nargs="?",
        help="load from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-n",
        "--n_samples",
        type=int,
        nargs="?",
        help="number of samples to draw",
        default=50000
    )
    parser.add_argument(
        "--dataroot",
        type=str,
        nargs="?",
        help="data root",
        default='/path/to/data'
    )
    parser.add_argument(
        "--outdir_name",
        type=str,
        nargs="?",
        help="output directory",
        default='LDMDTI'
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        nargs="?",
        help="the bs",
        default=48
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        nargs="?",
        help="number of workers",
        default=0
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="start index for dataset slicing",
    )
    parser.add_argument(
        "--end_idx",
        type=int,
        default=None,
        help="end index for dataset slicing",
    )
    parser.add_argument("--seed", type=int, default=42, help="frozen base random seed")
    parser.add_argument("--model_id", type=str, default="ldm3d", help="stable model identifier")
    # opt = parser.parse_args()
    opt, unknown = parser.parse_known_args()
    
    if not os.path.exists(opt.resume):
        raise ValueError("Cannot find {}".format(opt.resume))
    if os.path.isfile(opt.resume):
        paths = opt.resume.split("/")
        idx = -2  # take a guess: path/to/logdir/checkpoints/model.ckpt
        logdir = "/".join(paths[:idx])
        ckpt = opt.resume
    else:
        assert os.path.isdir(opt.resume), f"{opt.resume} is not a directory"
        logdir = opt.resume.rstrip("/")
        ckpt = os.path.join(logdir, "model.ckpt")
        
    base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
    opt.base = base_configs

    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)

    if opt.batch_size != 1:
        raise ValueError("External inference requires --batch_size 1")
    if opt.n_samples != 1:
        raise ValueError("External inference requires --n_samples 1")

    test_transform = get_val_transforms_3d()
    test_dataset = PairedVolumeSample(dataroot=opt.dataroot, phase='test', transform=test_transform, outdir_name=opt.outdir_name, steps=opt.steps, start_idx=opt.start_idx, end_idx=opt.end_idx)
    if len(test_dataset) == 0:
        print("No new data to process. Exiting.")
        sys.exit()
    test_loader = DataLoader(test_dataset, batch_size=opt.batch_size,
                          num_workers=opt.num_workers, shuffle=False)
    
    gpu = True
    eval_mode = True
    model, global_step = load_model(config, ckpt, gpu, eval_mode)
    sampler = DDIMSampler(model)
    sampler_kwargs = {'eta': 0.0}
    shape = (model.channels, *model.image_size)
    temporal_length = model.temporal_length
    sampling_batch_size = opt.batch_size * temporal_length
    
    with torch.no_grad():
        with model.ema_scope():
            for test_batch in tqdm(test_loader):
                patient_id, study_date, series_id = test_batch['patient_id'], test_batch['study_date'], test_batch['series_id']
                current_seed = stable_case_seed(
                    opt.seed, opt.model_id, patient_id[0], study_date[0], series_id[0]
                )
                set_inference_seed(current_seed)
                
                mask = test_batch['mask']
                lesion = test_batch['lesion']
                
                dwi_mean = test_batch['dwi_mean']
                dwi_std = test_batch['dwi_std']
                adc_mean = test_batch['adc_mean']
                adc_std = test_batch['adc_std']
                
                z, c, mask, lesion, x, xrec, xc = model.get_input(test_batch,
                                                                model.first_stage_key,
                                                                return_first_stage_outputs=True,
                                                                force_c_encode=True,
                                                                return_original_cond=True,
                                                                bs=opt.batch_size)
                
                samples, intermediates = sampler.sample(opt.steps, ddim_use_original_steps=False, batch_size=sampling_batch_size, shape=shape, conditioning=c, verbose=False, **sampler_kwargs)

                x_samples = model.decode_first_stage(samples)
                
                x_samples = normalize_data(x_samples.detach().cpu(), dwi_mean, dwi_std, adc_mean, adc_std)
                x_samples = x_samples * mask
                xrec = normalize_data(xrec.detach().cpu(), dwi_mean, dwi_std, adc_mean, adc_std)
                xrec = xrec * mask
                
                x_samples = rearrange(x_samples, '(b d) c h w -> b c h w d', d=temporal_length)
                xrec = rearrange(xrec, '(b d) c h w -> b c h w d', d=temporal_length)
                
                for i in range(x_samples.shape[0]):
                    save_result(
                        opt.dataroot,
                        opt.outdir_name,
                        opt.steps,
                        patient_id[i],
                        study_date[i],
                        series_id[i],
                        x_samples[i].numpy(),
                    )
        
