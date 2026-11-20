# EgoPoseVQVAE

Egocentric 3D human pose estimation from a single fisheye head-mounted-camera image. The repository contains two parts:

1. **SelfPose**: a ResNet-50 based network that predicts 2D joint heatmaps from the image and regresses 3D joint positions from them. It is trained on the xR-EgoPose dataset.
2. **Pose VQ-VAE**: a VQ-VAE trained on AMASS poses (converted to the Human3.6M joint layout through SMPL). It is used as a learned pose prior to refine the SelfPose prediction; `test_egopose_vqvae.py` compares SelfPose, SelfPose + VQ-VAE and ground truth.

## Pipeline

```
image --> SelfPose encoder --> 2D heatmaps --> SelfPose decoder/regressor --> 3D joints
                                                                                 |
                                     (optional)  pose VQ-VAE (encode, quantise, decode) --> refined 3D joints
```

## Repository layout

```
configs/        one YAML per experiment (model1..6.yml for SelfPose, vqvae1..2.yml for the VQ-VAE)
base/           base dataset / transform / evaluation classes (from xR-EgoPose)
dataset/        mocap.py (xR-EgoPose), amass.py + smpl.py (AMASS via SMPL), transform.py
model/          self_pose.py, resnet.py (SelfPose); pose_VQVAE.py, encdec.py, quantize_cnn.py,
                resnet1D.py, embedder.py (VQ-VAE)
utils/          config.py, runtime.py, visualize.py, evaluate.py, io.py, console_logger.py
data/Human36M/  J_regressor_h36m_correct.npy (tracked; SMPL -> H36M joint regressor)
train.py  test.py  train_vqvae.py  test_vqvae.py  test_egopose_vqvae.py
```

## Installation

```
pip install -r requirements.txt
git clone https://github.com/gulvarol/smplpytorch && pip install ./smplpytorch   # AMASS / VQ-VAE only
```

Install `torch` and `torchvision` first according to your CUDA setup. `smplpytorch` is only imported by `dataset/smpl.py`, so SelfPose training and testing do not need it.

## Data preparation

Default locations are set in the YAML configs and can be overridden on the command line.

**xR-EgoPose** (<https://github.com/facebookresearch/xR-EgoPose>), default `./data/xr-egopose`:

```
data/xr-egopose/{TrainSet,ValSet,TestSet}/.../rgba/*   frames
                                         .../json/*    per-frame annotations
```

`dataset/mocap.py` walks each split recursively and indexes every directory that contains both `rgba/` and `json/`; the nesting above them is arbitrary.

**AMASS** (<https://amass.is.tue.mpg.de>), default `./data/AMASS`:

```
data/AMASS/<SubDataset>/<sequence>/*_poses.npz
```

`dataset/amass.py` globs `<root>/*/*/*_poses.npz` and reads the `poses`, `betas` (and `dmpls`, `trans`, `gender`) arrays from each file, so the SMPL+H/DMPL-style `.npz` files are expected. The first 72 pose values and 10 shape coefficients are used. Frames are subsampled per sub-dataset (every 5th by default; the test split uses every 60th).

**SMPL model** (<https://smpl.is.tue.mpg.de>), default `./data/smplpytorch/native/models`: the model files expected by smplpytorch (<https://github.com/gulvarol/smplpytorch>); the code builds the `neutral` layer.

## Experiments

All SelfPose experiments are defined in `configs/`; the YAML files are the source of truth.

| config | skeleton (joints) | variant | center crop | heatmap | rotation head | batch |
|---|---|---|---|---|---|---|
| model1 | 16 | full (ResNet-50 to layer4, 2048-d) | no | 51 | no | 12 |
| model2 | 16 | light (to layer3, 1024-d, shallower decoders) | yes | 47 | no | 24 |
| model3 | 22 | light | yes | 47 | no | 24 |
| model4 | 16 | full | yes | 51 | yes | 12 |
| model5 | 14 (no toes) | full | yes | 51 | yes | 12 |
| model6 | 14 | full | yes | 51 | no | 12 |

`vqvae1` uses positional Fourier features on the input pose, `vqvae2` uses none. `model6` + `vqvae1` is the pair used by `test_egopose_vqvae.py` in the examples below.

## Usage

Outputs go to `./ckpts/<name>/` (checkpoints) and `./train_logs/<name>.log`, as set in the config.

```
# train SelfPose
python train.py --config configs/model6.yml --data-root /data/xr-egopose --gpu 0

# visualise SelfPose predictions against ground truth (--split test|val|train, default test)
python test.py --config configs/model6.yml --checkpoint ckpts/model6/<file>.pth --split test --gpu 0

# train the pose VQ-VAE on AMASS (--resume PATH to continue)
python train_vqvae.py --config configs/vqvae1.yml --amass-root /data/AMASS --smpl-root /data/smplpytorch/native/models --gpu 0

# visualise VQ-VAE reconstructions of AMASS or xR-EgoPose ground-truth poses
python test_vqvae.py --config configs/vqvae1.yml --checkpoint ckpts/model_vqvae1/<file>.pth --data amass
python test_vqvae.py --config configs/vqvae1.yml --checkpoint ckpts/model_vqvae1/<file>.pth --data xr-egopose \
    --egopose-config configs/model6.yml --data-root /data/xr-egopose

# SelfPose vs SelfPose + VQ-VAE vs ground truth
python test_egopose_vqvae.py --config configs/model6.yml --checkpoint ckpts/model6/<file>.pth \
    --vqvae-config configs/vqvae1.yml --vqvae-checkpoint ckpts/model_vqvae1/<file>.pth
```

`train.py` and `train_vqvae.py` additionally accept `--epochs`, `--batch-size` and `--output-dir DIR` (write checkpoints and the log under `DIR` instead of the config's `output` paths). Without `--gpu` the first visible GPU is used, or the CPU if CUDA is unavailable. The test scripts open one matplotlib figure per sample.

## Configuration

Annotated `configs/model6.yml`:

```yaml
name: model6
dataset:
  root: ./data/xr-egopose    # contains TrainSet/ ValSet/ TestSet/
  train: TrainSet
  val: ValSet
  test: TestSet
  center_crop: true          # crop central square before resizing to 368x368 (false: resize full frame)
  heatmap_size: 51           # side of the ground-truth heatmaps
  rotation: false            # dataset additionally returns joint rotations
skeleton: '14'               # '14' | '16' | '22' (see utils/config.py)
transforms:
  norm: Neck                 # root joint the 3D pose is normalised around
model:
  variant: full              # full | light
  rotation: false            # add the rotation head
data_loader: {batch_size: 12, shuffle: true}
val_data_loader: {batch_size: 1, shuffle: true}
train: {epochs: 7, lr: 0.0001, lr_gamma: 0.95, save_every: 5000, heatmap_div: 50}   # loss = pose + (hm + hm2) / heatmap_div
output: {ckpt_dir: ./ckpts/model6, log_file: ./train_logs/model6.log}
eval: {output_file: ./eval_res.json}
```

The VQ-VAE configs have an `amass` block (`root`, `smpl_model_root`), a `model` block (`down_t`, `fourier_features: positional | 'none'`) and a `train` block (optimiser, `MultiStepLR` milestones, `commit_weight`). To add an experiment, copy a YAML, change `name` and the `output` paths, and edit the values.

## Notes

- `model3` is root-normalised around `Hips`, not `Neck`: the original run looked the root joint index up in the 16-joint skeleton. See the comment in `configs/model3.yml`.
- In experiments 4 and 5 the rotation loss term was computed on the pose tensors rather than on the predicted/ground-truth rotations, so the rotation head received no supervision. `train.py` preserves this for reproducibility.
- Checkpoints trained with the pre-refactor code load into the unified `SelfPose` model unchanged (state_dict keys and shapes are identical, verified for all six variants).
- The original `test.py` and `test2.py` (experiments 1 and 2) visualised the **training** split; pass `--split train` to reproduce that. The default is now `test` for every experiment.
- The SelfPose encoder is built directly from the `ResNet` class in `model/resnet.py`, so no ImageNet weights are loaded; the backbone is trained from scratch.
- `utils/evaluate.py` (from xR-EgoPose) is not called by any script and its upper/lower-body joint index lists assume the 16-joint skeleton.

## Acknowledgements

- SelfPose code structure, `base/`, `utils/` and `dataset/mocap.py` derive from [facebookresearch/xR-EgoPose](https://github.com/facebookresearch/xR-EgoPose) (Facebook copyright headers retained in the files).
- `dataset/amass.py` and `dataset/smpl.py` are adapted from Pose2Mesh_RELEASE by hongsukchoi.
- `model/pose_VQVAE.py`, `encdec.py`, `quantize_cnn.py` and `resnet1D.py` are from T2M-GPT by Mael-zys.
- `model/embedder.py` is taken from [nerf-pytorch](https://github.com/yenchenlin/nerf-pytorch) / [nerf](https://github.com/bmild/nerf).
- `model/resnet.py` is based on the torchvision ResNet implementation.
- SMPL layer: [smplpytorch](https://github.com/gulvarol/smplpytorch).
