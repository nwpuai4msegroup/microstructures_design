# Model Card: Physics-Aware Generative Design of Microstructures

## Model summary

This repository implements the physics-aware generative-optimization framework described in **“Generative design of high-fidelity microstructures using physics-aware machine learning.”** The framework is designed to generate high-fidelity electron backscatter diffraction (EBSD) microstructure representations and to explore microstructures with targeted strength-ductility combinations.

The main workflow combines:

- a variational autoencoder (VAE) for learning a compact latent representation of microstructure images;
- a conditional denoising diffusion probabilistic model (DDPM), with a U-Net denoiser, for refining generated microstructures;
- physics-aware reconstruction losses that emphasize pixel similarity, grain boundaries, and crystallographic structure;
- principal component analysis (PCA) and property-prediction surrogate models for relating latent descriptors to yield strength and elongation; and
- the NSGA-II multi-objective evolutionary algorithm for exploring the latent space and identifying candidate microstructures with improved strength-ductility synergy.

Additional models are used to prepare generated microstructures for crystal plasticity finite element (CPFE) validation, including a fine-tuned ResNet-18 model for geometrically necessary dislocation (GND) density prediction and a U-Net model for grain-boundary removal.

## Model details

| Item | Description |
| --- | --- |
| Developed by | Weijie Liao, Ruihao Yuan, and collaborators at Northwestern Polytechnical University |
| Institution | State Key Laboratory of Solidification Processing, Northwestern Polytechnical University |
| Model type | Physics-aware VAE and conditional DDPM, coupled with property surrogates and NSGA-II |
| Application domain | Data-driven design of metallic microstructures |
| Material system demonstrated | Inconel 625 nickel-based superalloy |
| Inputs | EBSD-derived microstructure images represented through reversible mappings of three Euler angles to RGB channels |
| Main outputs | Reconstructed or generated microstructure images and optimized latent descriptors associated with predicted yield strength and elongation |
| Repository | https://github.com/nwpuai4msegroup/microstructures_design |
| Archived release | https://doi.org/10.5281/zenodo.21897025 |
| Contact | Ruihao Yuan (rhyuan@nwpu.edu.cn); Jinshan Li (ljsh@nwpu.edu.cn) |

## Intended use

### Primary intended uses

The framework is intended for research on:

- reconstruction and generation of crystallographic microstructure images;
- physics-aware evaluation of grain morphology, grain boundaries, and lattice-orientation fidelity;
- latent-space exploration and multi-objective inverse design of microstructures;
- hypothesis generation for processing-structure-property studies; and
- selection of candidate microstructures for subsequent physics-based simulation and experimental validation.

The released implementation is primarily a research artifact supporting reproducibility of the associated study. It may also provide a starting point for adapting the workflow to other alloy systems when suitable material-specific data and validation procedures are available.

### Out-of-scope uses

The models should not be used as:

- a substitute for experimental characterization or physics-based validation;
- a production-quality predictor for alloy systems, processing windows, imaging modalities, or property ranges not represented in the training data;
- an autonomous system for selecting manufacturing conditions without expert review; or
- a safety-critical decision tool.

Predictions for out-of-distribution microstructures should be treated as candidate hypotheses until they have been evaluated using independent simulations and experiments.

## Training data

The experimental dataset contains 25 Inconel 625 variants produced through an orthogonal thermo-mechanical processing design. Each condition is associated with an EBSD microstructure, yield strength, and elongation. The processing matrix varies hot-rolling reduction, heat-treatment temperature, and heat-treatment duration.

The original EBSD Euler maps have a resolution of 1000 × 800 pixels and a pixel size of 0.5 μm. The three Euler angles are mapped reversibly to the red, green, and blue image channels. Each original image is uniformly cropped into 80 patches of 100 × 100 pixels, yielding 2,000 patches before augmentation.

The image-generation training data are expanded using horizontal and vertical flips, rotations, Gaussian blurring, distortion, black-block occlusion, erosion, and dilation. Degraded images are used as inputs with their corresponding original images as restoration targets. The complete augmentation procedure yields 72,000 image patches: 12,000 for reconstruction and 60,000 for restoration.

The dataset is intentionally small at the level of independent alloy conditions but covers diverse thermo-mechanical conditions. Augmented patches increase the number of training examples but do not constitute additional independent experimental specimens.

## Data splitting and leakage control

The 25 original high-resolution microstructures are randomly divided into an 80% training set and a 20% held-out test set. Cropping and augmentation are performed within the resulting partitions so that patches derived from the same original microstructure are not used across training and testing. This original-image-level split is important for reducing leakage between highly related image patches.

The property-prediction surrogate models are evaluated on a held-out 20% subset of the 25 experimental conditions. The GND-prediction model uses an 80/20 training/test split of the cropped experimental microstructures.

Because the number of independent experimental conditions is limited, the study does not report a separate third validation partition or k-fold cross-validation. Model assessment therefore combines held-out testing with physical-feature evaluation, comparisons against generative baselines, CPFE simulations, and validation of a newly fabricated microstructure.

## Preprocessing

The principal preprocessing steps are:

1. EBSD acquisition and processing using Channel 5 software;
2. reversible mapping of Euler1, Euler2, and Euler3 values to RGB channels;
3. uniform patch extraction from each high-resolution microstructure;
4. geometric and degradation-based image augmentation; and
5. conversion of selected generated microstructures into crystallographic-orientation and GND inputs for CPFE simulation.

Users should follow the repository scripts and configuration files to reproduce the precise tensor transformations and model input preparation used in the released implementation.

## Architecture and training

### Physics-aware VAE

The VAE encoder uses convolutional layers to map microstructure images to 128-dimensional probabilistic latent variables parameterized by a mean and standard deviation. The decoder uses transposed convolutional layers to reconstruct the microstructure images.

The VAE objective combines:

- pixel-wise mean absolute error, \(L_{\mathrm{pix}}\);
- Sobel-based edge loss, \(L_{\mathrm{edge}}\), which emphasizes grain boundaries and interfaces;
- structural-similarity loss, \(L_{\mathrm{SSIM}}\), which captures local structural and crystallographic consistency; and
- Kullback-Leibler divergence, \(L_{\mathrm{KL}}\), for latent-space regularization.

### Conditional DDPM

The DDPM refines image details using a U-Net denoiser conditioned on the VAE output. Its training loss includes pixel, SSIM, and edge components applied to the predicted and target noise representations.

### Property surrogates and inverse design

For each high-resolution microstructure, the latent variables of its 80 patches are summarized using mean and standard-deviation descriptors. PCA is applied separately to these descriptors, and the first three principal components from each are retained, resulting in a six-dimensional representation.

Surrogate regression models predict yield strength and elongation from these latent descriptors. Their predictions are used as the two objectives in NSGA-II. The reported optimization uses a population size of 10 and 500 generations.

### Training configuration

The image-generation models are trained with the Adam optimizer and cosine-annealing learning-rate scheduling. The learning rate decreases from 0.0001 to zero over 100 epochs, with a batch size of 50.

## Evaluation

The framework is evaluated using complementary image, physical, property-prediction, and validation criteria.

### Image and physical fidelity

- pixel-wise reconstruction loss;
- edge loss;
- structural similarity (SSIM);
- Fréchet distance (FD) computed from Inception-v3 feature representations;
- grain area;
- major and minor axes of equivalent grain ellipses;
- Euler1, Euler2, and Euler3 distributions; and
- crystallographic misorientation distributions.

The physical descriptors are evaluated for both training and held-out test microstructures. Statistics from patches belonging to the same original image are also aggregated to the original-image level.

### Property prediction and optimization

- held-out test-set coefficient of determination, \(R^2\), for yield-strength and elongation surrogate models;
- hypervolume improvement for tracking multi-objective optimization; and
- agreement between predicted target properties, CPFE results, and experimental tensile measurements.

The yield-strength and elongation surrogate models each achieve a reported test-set \(R^2\) above 0.92. The GND-prediction model achieves a reported test-set \(R^2\) of 0.90.

### Baselines and ablations

The proposed model is compared with:

- a reconstruction model trained using pixel-wise loss alone;
- a standard VAE;
- a Wasserstein generative adversarial network (WGAN); and
- a standard DDPM.

The study also evaluates the contribution of the edge and SSIM loss terms and examines how the number of retained PCA components affects reconstruction fidelity.

### Independent physical validation

Candidate microstructures are evaluated using a calibrated CPFE framework. A newly fabricated bimodal microstructure, absent from the training dataset, is characterized by EBSD and tensile testing. Its observed mechanical response is compared with both the machine-learning prediction and CPFE simulation. The GND-prediction model is additionally evaluated on this out-of-distribution experimental microstructure.

## Limitations and potential biases

- **Limited number of independent experimental conditions.** Although augmentation produces many patches, the dataset originates from 25 independent alloy-processing conditions. Patch counts should not be interpreted as the number of independent specimens.
- **Material and acquisition specificity.** The models are demonstrated on Inconel 625 EBSD data collected and processed under a consistent protocol. Performance may change for other materials, microscopes, scan settings, resolutions, or preprocessing pipelines.
- **Training-distribution bias.** Generative models tend to reproduce the available training distribution. Physics-aware losses, latent-space optimization, and independent validation reduce this risk but do not eliminate it.
- **Limited uncertainty quantification.** The reported surrogate predictions are used to guide candidate selection, but they should not be interpreted as calibrated predictive uncertainty bounds.
- **Two-dimensional representation.** The generated EBSD maps describe two-dimensional sections and may not capture all three-dimensional microstructural characteristics.
- **Finite physical descriptors.** Grain morphology and Euler-angle metrics provide important physical checks but do not exhaustively characterize microstructure fidelity.
- **No formal k-fold cross-validation.** Evaluation relies on a held-out test split and complementary physics-based and experimental validation.
- **Application-dependent synthesizability.** A generated microstructure is not guaranteed to be experimentally achievable solely because it lies in a plausible region of the learned latent space.

## Responsible use recommendations

Users should:

1. retain independent experimental conditions when creating training and test partitions;
2. prevent patches or augmented variants from the same original image from crossing partitions;
3. report results at both patch and independent-specimen levels;
4. assess generated microstructures using domain-relevant physical descriptors rather than image similarity alone;
5. validate promising out-of-distribution candidates through appropriate physics-based simulations and experiments; and
6. retrain and revalidate the models before applying them to a different material system or imaging protocol.

No human-subject, personal, demographic, or other sensitive data are used in this project.

## Reproducibility and availability

Source code and reproduction materials are available from:

- GitHub: https://github.com/nwpuai4msegroup/microstructures_design
- Zenodo: https://doi.org/10.5281/zenodo.21897025

Data underlying the manuscript figures are provided in the manuscript, Supplementary Information, and Source Data files. Additional data may be requested from the corresponding authors.

The manuscript reports the model architectures, optimizer, learning-rate schedule, number of epochs, batch size, data augmentation, evaluation metrics, and validation workflow. Hardware specifications, wall-clock training time, energy use, and carbon-emission estimates are not reported and may vary with implementation and computing environment.

The pretrained ResNet-18 backbone used for GND prediction is available through TorchVision. Users should consult the applicable TorchVision and pretrained-weight terms when redistributing or reusing those weights.

## License

This model card does not itself grant a software, model-weight, or data license. Use and redistribution are governed by the license and terms included with the corresponding GitHub or Zenodo release. Third-party components and pretrained weights remain subject to their respective licenses and terms.

## Citation

If you use this code, model, or associated data, please cite the archived release:

```bibtex
@software{liao2026physics_aware_microstructure,
  author    = {Liao, Weijie and Li, Kaidi and Tang, Bin and Fan, Jiangkun and Wang, Jun and Xue, Xiangyi and Li, Jinshan and Yuan, Ruihao},
  title     = {Generative design of high-fidelity microstructures using physics-aware machine learning},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.21897025},
  url       = {https://doi.org/10.5281/zenodo.21897025}
}
```

When the associated journal article is published, users should also cite the final article record.

## Model-card version

- Model-card version: 1.0
- Last updated: 2026-09-06
- Associated manuscript: NCOMMS-26-007696B

