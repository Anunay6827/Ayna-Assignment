# Polygon Color Fill – Report & Insights

## A Dual Encoder based approach 
## 1. Hyperparameters
### Tried
- **Learning Rate:** Started at `1e-3`, tried `5e-4`, `1e-4`. Higher LR made early convergence faster but increased instability in color prediction.
- **Batch Size:** Tested `8`, `16`, and `32`. `16` gave best balance between GPU memory use and stability.
- **Loss Functions:**
  - Pure `L1Loss` (baseline)
  - `L1Loss + MSE color loss` inside polygon mask (**final choice**)
  - Tried `SSIM` addition but no noticeable improvement.
- **COLOR_LOSS_WEIGHT:** Tried `1.0`, `2.0`, `5.0`; settled on `2.0` for better hue matching without degrading shape.
- **Augmentation:** Random rotation (±30°), horizontal flip; improved robustness to unseen orientations.

### Final Settings
| Hyperparameter         | Value       |
|------------------------|-------------|
| Learning Rate          | `1e-3`      |
| Batch Size             | `16`        |
| Epochs                 | `30`        |
| Optimizer              | Adam        |
| Loss                   | `L1 + 2.0 × Color MSE` |
| Augmentation           | Rotation, Flip |
| Scheduler              | ReduceLROnPlateau (factor=0.5, patience=3) |

## 2. Architecture
### Base
- U-Net from scratch in PyTorch.
- **Dual Encoder:**
  - **Polygon Encoder:** Standard CNN encoder (U-Net down path) for grayscale polygon image.
  - **Color Encoder:** 
    - Learns an embedding vector for the color name.
    - Produces both a small spatial color feature map (for bottleneck concatenation) and an embedding vector (for FiLM).

### Conditioning
- **Bottleneck Fusion:** Concatenate polygon bottleneck features (512 channels) with color spatial map (256 channels), then reduce to 512 channels.
- **FiLM Modulation:** Apply Feature-wise Linear Modulation at **every decoder block**, using the color embedding vector to scale and shift features. This strengthened color conditioning and fixed the “yellow/orange collapse.”

## 3. Training Dynamics
- **Early epochs:** Model quickly learned polygon boundaries and started filling shapes, but colors were biased toward warm tones.
- **After FiLM + color loss:** Color accuracy improved significantly, with predictions matching requested hues within ~5 epochs.
- **Loss Curves:** 
  - Recon loss dropped steadily.
  - Color loss dropped sharply after epoch 5, then plateaued.
- **Qualitative Trends:**
  - Shapes became crisp early on.
  - Color fidelity improved later once FiLM was applied consistently.

## 4. Failure Modes & Fixes
| Failure Mode | Cause | Fix |
|--------------|-------|-----|
| Warm color bias (yellow/orange) | Weak conditioning; model defaulted to dataset mean | Added FiLM at all decoder levels; added color loss |
| Blurry edges on small polygons | Downsampling loss in U-Net | Increased base filters from 32 to 64 |
| Overfitting to dominant colors | Imbalanced dataset | Augmentation; considered weighted sampling |

## 5. Key Learnings
- **Conditioning must be injected multiple times** — single-point color conditioning at bottleneck was not enough; multi-level FiLM was crucial.
- **Color-specific loss** guided the network to respect hue requirements more than L1 alone.
- **Data augmentation** improved generalization to unseen polygon sizes/orientations.
- **Monitoring both reconstruction loss and semantic metrics** (like mean color accuracy) was essential for debugging.
- Even simple shapes benefit from robust conditioning mechanisms when the output is strongly dependent on a non-visual input (color name).

## Links
- **Wandb Run** - https://wandb.ai/anunay6827-mahindra-university/dual-unet-polygons?nw=nwuseranunay6827
- **Model Link** - https://drive.google.com/file/d/1uWWEQTBkTYidgGIkg28FNJysYiTOXAXQ/view?usp=sharing
