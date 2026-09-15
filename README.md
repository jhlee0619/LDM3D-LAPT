# LDM3D-LAPT

Synthesising DWI and ADC maps from CT perfusion in acute ischaemic stroke with a
3D latent diffusion model and **lesion-aware post-training (LAPT)**. 3D conversion
layers give a pretrained 2D model inter-slice context; post-training then adds a
whole-image term, a lesion-weighted term and a term supervised by a frozen
segmentation network, so lesion voxels survive latent compression.

Junhyeok Lee, Hyunwoong Kim, Hyungjin Chung, Heeseong Eom, Han Jang, Joon Jang,
Chul-Ho Sohn, Kyu Sung Choi.

Extends our MICCAI 2025 paper, whose 2D code is
[snuh-rad-aicon/Diffusion-LAPT](https://github.com/snuh-rad-aicon/Diffusion-LAPT).
Training and sampling only — no evaluation, preprocessing, weights or data.

## Installation

```bash
conda create -n ldm3d-lapt python=3.9 && conda activate ldm3d-lapt
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

`numpy<2` is required (MONAI 1.3.2 breaks on 2.x). VGG-LPIPS weights are
downloaded on first use.

## Data

Not distributable. Volumes are 224 × 224 × 32 at 1 × 1 × 4 mm³, with 15 CTP time
points stacked as channels.

```
<dataroot>/
├── nifti/<patient_id>/<study_date>/<series_id>/
│   ├── ctp.nii.gz  dwi.nii.gz  adc.nii.gz
│   ├── ctp_mask.nii.gz   # brain mask
│   └── lesion.nii.gz     # training target
└── splits/
    ├── train_list.json  test_list.json
    └── slice_index.json  # usable slices, for the slice-wise stages
```

## Training

Five stages, in order; each loads the earlier checkpoints from `checkpoints/`.

```bash
DATAROOT=/path/to/data
run () { python train.py --base "$@" -t --gpus 0, --project ldm3d-lapt \
           data.params.train.params.dataroot=$DATAROOT \
           data.params.validation.params.dataroot=$DATAROOT; }

run configs/autoencoder/vqgan.yaml             # -> autoencoder.ckpt
run configs/segmentation/segmentation_3d.yaml  # -> segmentation_3d.ckpt (frozen guide)
run configs/latent-diffusion/ldm_2d.yaml       # -> ldm_2d.ckpt          (initialisation)
run configs/latent-diffusion/ldm_3d.yaml       # -> ldm_3d.ckpt          (+ 3D conversion)
run configs/latent-diffusion/ldm_3d_lapt.yaml  # full model              (+ LAPT)
```

Paper loss weights are the defaults in `ldm_3d_lapt.yaml` (`img 0.01`,
`lesion 0.0001`, `seg 0.001`); override as `model.params.seg_loss_weight=...`.

Conditioner ablations add `nocond.yaml` as a second config. Both paths go under
one `--base`: repeating the flag replaces the list rather than appending.

```bash
run configs/latent-diffusion/ldm_3d.yaml      configs/latent-diffusion/nocond.yaml
run configs/latent-diffusion/ldm_3d_lapt.yaml configs/latent-diffusion/nocond.yaml
```

Logging is local only — TensorBoard scalars and sample PNGs under `logs/<run>/`.

## Sampling

Architecture is restored from the config snapshot next to the checkpoint, so one
command covers the full model and both ablations. The paper uses 250 DDIM steps.

```bash
python sample.py -r logs/<run>/checkpoints/last.ckpt \
    --dataroot /path/to/data --outdir_name my_output --steps 250
```

## Citation

The journal article is under review; please cite the MICCAI 2025 paper.

```bibtex
@inproceedings{lee2025lesion,
  title     = {Lesion-Aware Post-training of Latent Diffusion Models for
               Synthesizing Diffusion MRI from CT Perfusion},
  author    = {Lee, Junhyeok and Kim, Hyunwoong and Chung, Hyungjin and
               Eom, Heeseong and Jang, Joon and Sohn, Chul-Ho and Choi, Kyu Sung},
  booktitle = {MICCAI},
  pages     = {282--291},
  year      = {2025},
  doi       = {10.1007/978-3-032-04937-7_27}
}
```

## Acknowledgements and license

MIT — see [LICENSE](LICENSE). Derived from CompVis'
[latent-diffusion](https://github.com/CompVis/latent-diffusion) and
[taming-transformers](https://github.com/CompVis/taming-transformers), with the
diffusion schedule and UNet backbone from OpenAI's
[improved-diffusion](https://github.com/openai/improved-diffusion) and
[guided-diffusion](https://github.com/openai/guided-diffusion), and 3D conversion
following [Make-A-Volume](https://doi.org/10.1007/978-3-031-43999-5_70).
Per-file attribution is in [NOTICE.md](NOTICE.md).
