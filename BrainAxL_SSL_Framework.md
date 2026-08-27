# BrainAxL: Self-Supervised Learning Framework and Architecture

**Proposed paper-facing model name:** **BrainAxL**  
**Expanded name:** **Brain Axial LSTM**  
**Suggested full title:** *BrainAxL: A Multiscale Axial LSTM Foundation Model for 3D Brain MRI*

## Model class

**Selected option:** Other

**Explanation:** Hybrid 3D CNN–LSTM architecture consisting of a U-Net-style convolutional encoder augmented with bidirectional axial LSTM blocks. The architecture does not use Transformer attention or state-space blocks.

## Encoder architecture

The encoder is a five-stage hybrid 3D CNN–LSTM architecture based on a U-Net-style hierarchy. For a 64×64×64 input patch, the five stages produce feature maps at resolutions of 64³, 32³, 16³, 8³, and 4³ voxels, with 40, 80, 160, 320, and 640 channels, respectively. Each stage contains two consecutive 3×3×3 convolutions with stride 1 and padding 1. Each convolution is followed by dropout configured with p=0, instance normalization, and LeakyReLU activation with a negative slope of 0.01. Downsampling is performed using four 2×2×2 max-pooling operations with stride 2.

Bidirectional axial LSTM blocks are applied at the two deepest stages: 320 channels at 1/8 resolution and 640 channels at 1/16 resolution. At each of these stages, three independent bidirectional LSTMs process the feature map along the height, width, and depth axes. Each LSTM has an input size equal to the stage width and a hidden size equal to half that width per direction, preserving the stage width after concatenating the two directions. The three axis-wise outputs are averaged, normalized using single-group GroupNorm, modulated by a learned 1×1×1 convolution followed by a sigmoid gate, and added residually to the convolutional feature map.

The encoder returns all five feature scales for skip connections to the reconstruction decoder. During self-supervised pretraining, global average and maximum pooling are applied to every encoder scale. The pooled vectors are concatenated into a 2,480-dimensional multiscale representation and projected to 512 dimensions for VICReg-style regularization.

## 1. Encoder number of parameters

**31,837,640 trainable parameters** for the single-channel encoder.

## 2. Pretraining decoder/projector architecture

Pretraining used both a reconstruction decoder and a semantic projection head.

The reconstruction decoder is a four-stage 3D U-Net decoder. Starting from the deepest 640-channel encoder feature map, each stage uses a 2×2×2 transposed convolution with stride 2 for upsampling. The upsampled features are concatenated with the corresponding encoder skip features and processed by two consecutive 3×3×3 convolutions. The decoder channel progression is 640→320→160→80→40. Each convolution is followed by dropout configured with p=0, instance normalization, and LeakyReLU activation with a negative slope of 0.01. A final 1×1×1 convolution maps the 40-channel full-resolution feature map to a single-channel reconstructed MRI volume. No output activation or deep supervision was used during pretraining.

For semantic regularization, global average pooling and global maximum pooling are applied independently to all five encoder feature maps, whose channel dimensions are 40, 80, 160, 320, and 640. The pooled vectors are concatenated to form a 2,480-dimensional multiscale representation. The projector consists of LayerNorm over the 2,480-dimensional vector followed by a bias-free linear layer from 2,480 to 512 dimensions. An additional non-parametric layer normalization is applied to the projected representation before computing the VICReg-style loss.

## 3. Decoder/projector number of parameters

- Reconstruction decoder: **13,196,241 parameters**
- Semantic projector: **1,274,720 parameters**
- Decoder and projector combined: **14,470,961 parameters**

Projector breakdown:

- LayerNorm(2,480): 4,960 parameters
- Bias-free Linear(2,480→512): 1,269,760 parameters

## 4. Total number of parameters active during pretraining

**46,308,601 trainable parameters.**

This includes 31,837,640 encoder parameters, 13,196,241 reconstruction-decoder parameters, and 1,274,720 semantic-projector parameters. All three components were optimized jointly during self-supervised pretraining.

## 5. Pretraining objective

**Primary objective: masked image reconstruction/denoising autoencoding (MAE-style), combined with non-contrastive VICReg-inspired variance–covariance regularization.**

For each sample, a normalized single-channel 64×64×64 MRI crop was divided into a 16×16×16 grid of non-overlapping 4×4×4 voxel blocks, giving 4,096 blocks. A random 2,457 blocks were replaced with zero, corresponding to a realized masking fraction of $2457/4096\approx0.59985$ (approximately 60%). The corrupted dense volume—not a shortened sequence containing only visible tokens—was passed through the complete 3D encoder and U-Net reconstruction decoder. The decoder predicted a one-channel 64×64×64 volume corresponding to the clean, unmasked target crop. Because the input could also contain blur, bias-field, gamma, and noise corruptions while the target did not, the reconstruction task additionally acted as denoising autoencoding.

The reconstruction loss was mean squared error between the prediction and clean target. The configuration used `rec_loss_masked_only=false`, so the error was averaged over all voxels, including both masked and visible regions. Thus, the method is MAE-style masked image modeling but is not a canonical Transformer MAE: masked blocks were zero-filled rather than removed from the encoder input, and reconstruction supervision was applied to the full volume rather than only to masked blocks.

In parallel, encoder features from all five spatial scales were each summarized by global average pooling and global maximum pooling. Concatenating these summaries produced a 2,480-dimensional multiscale representation. This vector was normalized, projected through a bias-free linear layer to 512 dimensions, and layer-normalized again. Across the local training batch, a variance penalty encouraged every projected dimension to maintain a standard deviation of at least one, while an off-diagonal covariance penalty encouraged different representation dimensions to be decorrelated.

The representation regularizer was weighted by 0.01 relative to reconstruction, with internal variance and covariance coefficients of 1.0 and 0.04, respectively. Its effective contribution was therefore $0.01\mathcal{L}_{\mathrm{var}}+0.0004\mathcal{L}_{\mathrm{cov}}$. There was no warm-up of this weight.

This was not a contrastive-learning objective: it used no negative pairs, InfoNCE loss, or contrastive queue. It was also not the full paired-view VICReg objective. The VICReg invariance/similarity coefficient was zero, so no clean-target teacher forward pass or paired-view similarity loss was computed. Teacher/global distillation and dense feature-consistency losses were disabled. Consequently, the only active objectives were full-volume reconstruction MSE and the variance–covariance regularization of the masked/corrupted-input representation.

## 6. Loss function

The total pretraining loss was:

$$
\mathcal{L}_{\mathrm{total}}
=
\mathcal{L}_{\mathrm{reconstruction}}
+
0.01\left(
\mathcal{L}_{\mathrm{variance}}
+
0.04\mathcal{L}_{\mathrm{covariance}}
\right).
$$

Reconstruction used mean squared error over the complete reconstructed volume:

$$
\mathcal{L}_{\mathrm{reconstruction}}
=
\frac{1}{N}
\sum_{i=1}^{N}
\left(\hat{x}_i-x_i\right)^2.
$$

Although 60% of the input blocks were masked, the checkpoint was configured with masked-only reconstruction disabled; therefore, MSE was averaged over all voxels, not only masked voxels.

For projected representations $Z\in\mathbb{R}^{B\times512}$, the variance term was:

$$
\mathcal{L}_{\mathrm{variance}}
=
\frac{1}{512}
\sum_{j=1}^{512}
\max\left(
0,\,
1-\sqrt{\operatorname{Var}(Z_j)+10^{-4}}
\right).
$$

After centering $Z$ across the batch, the covariance matrix was:

$$
C=\frac{(Z-\bar Z)^\top(Z-\bar Z)}{\max(B-1,1)},
$$

and the covariance penalty was:

$$
\mathcal{L}_{\mathrm{covariance}}
=
\frac{1}{512}
\sum_{i\neq j}C_{ij}^{2}.
$$

Equivalent pseudocode:

```text
reconstruction_loss = mean((reconstruction - target) ** 2)

std = sqrt(var(projected_features, dim=batch) + 1e-4)
variance_loss = mean(relu(1 - std))

centered = projected_features - mean(projected_features, dim=batch)
covariance = centered.T @ centered / max(batch_size - 1, 1)
covariance_loss = sum(off_diagonal(covariance) ** 2) / 512

total_loss = reconstruction_loss \
             + 0.01 * (variance_loss + 0.04 * covariance_loss)
```

## 7. Handling multiple sequences per session

MRI sequences were treated as independent, single-channel training samples. Each NIfTI scan was converted separately to a tensor with shape $1\times X\times Y\times Z$, irrespective of whether other sequences from the same imaging session were available. For example, T1w and T2w scans from the same session were presented as separate training instances rather than concatenated into a multi-channel input. No within-session registration, sequence fusion, or multi-encoder processing was performed during pretraining. Sampling frequency could vary through the coverage-aware sampling scheme, but sequence handling remained scan-level and single-channel.

## 2.3 Pretraining details

### Data augmentation

Each single-channel training volume first underwent volume-wise z-score normalization. A random 3D crop was then sampled with crop/pad handling. A larger intermediate crop was used before spatial augmentation, after which the sample was cropped to the final 64×64×64 voxel input size.

The spatial augmentation comprised random 3D rotation with probability 0.20, a conditional probability of 0.30 for rotation about each axis, and rotation magnitudes up to 10 degrees. Random isotropic scaling was applied with probability 0.20 using a scale factor sampled from 0.85 to 1.15. Elastic deformation was disabled.

The reduced-artifact GPU augmentation preset then independently applied Gaussian blur with probability 0.05 per channel, MRI bias-field augmentation with probability 0.10 per channel, gamma augmentation with probability 0.15 per sample, multiplicative noise with probability 0.05 per channel, and additive noise with probability 0.05 per channel. Motion ghosting, Gibbs ringing, and simulated low-resolution augmentation were not used in this preset.

The reconstruction target was copied after normalization and spatial augmentation but before GPU intensity corruption and masking, and target intensities were clamped to $[-2,4]$. Finally, 60% of the 4×4×4 voxel blocks in the input were randomly replaced with zero for masked reconstruction. This masking was part of the self-supervised input corruption rather than a geometric augmentation.

### Input patch size used

**1×64×64×64 voxels per sample**: one MRI channel and a spatial crop of 64×64×64 voxels.

### Batch size and accumulation steps used

**Batch size 128 on one GPU, with one gradient-accumulation step (`accumulate_grad_batches=1`).** The effective optimizer batch size was therefore **128**.

### Learning rate schedule

The initial learning rate was $1\times10^{-4}$. A cosine-annealing schedule was updated after every optimizer step for the complete 34,660-step run:

$$
\operatorname{lr}(t)
=
\frac{10^{-4}}{2}
\left[1+\cos\left(\frac{\pi t}{34660}\right)\right],
\qquad 0\leq t\leq34660.
$$

The minimum learning rate was zero. No learning-rate warm-up was actually used: both encoder/joint warm-up and decoder-only warm-up were set to zero epochs. A stored `warmup_ratio=0.02` field was inactive for this run.

### Optimizer and parameters

Training used fused AdamW with the following parameters:

```text
learning_rate = 1e-4
betas = (0.9, 0.98)
eps = 1e-8
weight_decay = 3e-5
amsgrad = false
fused = true
```

The same learning rate was applied to the encoder, reconstruction decoder, and semantic projector; no layer-wise learning rates were configured.

### Total training epochs and steps

**20 pseudo-epochs and 34,660 optimizer steps** for the selected checkpoint. Each pseudo-epoch contained 1,733 training steps and was followed by 60 validation batches.

### Epoch and step definition

An epoch was defined as **1,733 randomly sampled training iterations**, not as one complete dataset traversal. With batch size 128, the checkpoint-producing trainer drew **221,824 samples per pseudo-epoch** from a weighted training list containing 443,479 entries (290,897 unique scans). Sampling was with replacement, so scans could repeat while others were not selected; approximately 51.9% of unique scans were expected to appear in one pseudo-epoch. Across all 20 pseudo-epochs, 4,436,480 crops were drawn, making exposure to nearly every scan likely but not guaranteed. Sampling a scan also did not imply exhaustive spatial coverage because only one random 64×64×64 crop was generated per draw.

One step was one forward pass, backward pass, and AdamW parameter update using 128 crops on one GPU. Because gradient accumulation was one, every training iteration produced one optimizer update.

### Stopping criteria/checkpoint selection

Training used a fixed compute budget and stopped when `max_steps=34,660` was reached. There was no early stopping. A checkpoint was written after every pseudo-epoch to a fixed `last` checkpoint path, with `save_top_k=1` and no monitored validation metric. The final checkpoint at the end of the 34,660-step schedule was selected. It was not selected by best validation loss, linear probing, or downstream-task performance.

### Hyperparameter tuning

Hyperparameters were chosen manually through preliminary pilot runs and implementation defaults; no automated grid search, Bayesian optimization, or population-based search was used for this checkpoint. Pilot decisions were guided by GPU-memory feasibility and throughput, reconstruction behavior, and self-supervised validation diagnostics such as MSE, PSNR/SSIM, representation variance, covariance, and collapse statistics. After these checks, the patch size, masking ratio, optimizer settings, semantic-projection dimension, loss weights, and 20-pseudo-epoch budget were fixed for the reported run. Downstream labels were not used to select the pretraining checkpoint, and no post hoc linear-probe result was used for checkpoint selection.

## Computational Requirements

### Number of GPUs used

**1 GPU per pretraining run.**

### Type of GPUs used

**NVIDIA H100 80GB HBM3.**

The GPU reported 81,559 MiB of physical memory (approximately 79.6 GiB usable capacity). MIG was disabled.

### GPU memory used

**Approximately 56 GB.** A dummy-data replication of the pretraining workload used the complete 46.31-million-parameter encoder–decoder–projector, batch size 128, 64×64×64 inputs, bfloat16 mixed precision, full forward and backward propagation, and AdamW optimizer updates. PyTorch measured **52.40 GiB peak allocated memory** and **56.37 GiB peak reserved memory**. The rounded peak reserved value is reported as the practical VRAM requirement.

### Computational optimization methods

- **Automatic mixed-precision training using bfloat16 (`bf16-mixed`).**
- Single-GPU training.
- H100 Tensor Core-capable hardware.
- GPU-side data augmentation and masking.
- Eight CPU data-loader workers per training process.
- No gradient accumulation; `accumulate_grad_batches=1`.
- No model compilation; `compile_mode` was disabled.

### Other

- Local and effective optimizer batch size for the selected checkpoint: **128 samples**.
- Input crop size: **64×64×64 voxels**.
- Training steps: **34,660**.
- Wall-clock duration: **235,990 seconds**, approximately **65 hours 33 minutes**.
- Training framework: PyTorch 2.6.0 with CUDA 12.4 runtime and PyTorch Lightning.
