# Third-party code

This repository is a derivative work. The files below started as upstream code
and were modified for CT perfusion → DWI/ADC synthesis; the volumetric (3D)
variants, the lesion-aware post-training losses, the conditioner modules and the
dataset classes are our additions.

| Upstream | License | Where it lives here |
|---|---|---|
| [CompVis/latent-diffusion](https://github.com/CompVis/latent-diffusion) | MIT | `train.py`, `ldm3d_lapt/models/{ddpm,ddim,autoencoder}.py`, `ldm3d_lapt/modules/{unet,vqgan,attention,common,util,encoders,losses}.py` |
| [CompVis/taming-transformers](https://github.com/CompVis/taming-transformers) | MIT | `ldm3d_lapt/modules/{backbone_3d,quantize,discriminator,lpips,gan_losses}.py` |
| [openai/improved-diffusion](https://github.com/openai/improved-diffusion) | MIT | diffusion schedule and loss formulation inherited via the CompVis code (see the header of `ldm3d_lapt/models/ddpm*.py`) |
| [openai/guided-diffusion](https://github.com/openai/guided-diffusion) | MIT | UNet backbone in `ldm3d_lapt/modules/unet.py` |

Class names inside these files are unchanged from upstream, so they can still be
diffed against the original projects — but the files are a strict subset: classes
and helpers that nothing here references were deleted. Only project-specific names
were renamed.

## Our additions

- `ldm3d_lapt/models/ddpm_volume.py`, `ddpm_volume_lapt.py` — volumetric latent
  diffusion and the lesion-aware post-training objective
  (`img_loss_weight` + `lesion_loss_weight` + `seg_loss_weight`).
- `ldm3d_lapt/modules/unet_volume.py` — lightweight 3D conversion layers
  (`PseudoConv3d`) added to the 2D UNet.
- `ldm3d_lapt/modules/conditioners_3d.py` (`VolumeConditioner`) — the CTP conditioner.
- `ldm3d_lapt/models/segmentation.py` — the frozen segmentation network used for
  the guidance loss.
- `ldm3d_lapt/data/*.py` — CTP/DWI/ADC dataset classes.

## Restructuring for this release

The research tree kept two packages (`ldm/`, `taming/`) with the upstream nesting.
For this release they were merged into one package and the unused upstream code
(CLIP/text conditioning, the transformer stage of taming, BSRGAN degradations,
classifier guidance, PLMS) was removed rather than shipped dead, as were the
2D-only model variants that LDM3D-LAPT does not need. File and class
renames are limited to project-specific names; nothing upstream was renamed.
