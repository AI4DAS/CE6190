# Results

## 1. Pilot setup

The completed pilot compares full DreamBooth, DreamBooth-LoRA, Textual Inversion, and IP-Adapter Plus on the single non-human DreamBooth `dog2` subject. Each method generated 24 images from six semantic conditions and four predetermined seeds, for 96 images total. Generation used 512×512 resolution, 50 PNDM denoising steps, guidance scale 7.5, and FP16. IP-Adapter Plus used scale 0.7 and reference image `00.jpg`.

DINOv2 comparisons use only the six original dog2 reference images. CLIP comparisons use clean semantic prompts without the `sks` identifier or Textual Inversion placeholder token.

## 2. Subject fidelity

Mean whole-image DINOv2 similarity to all six references:

| Method | Mean | Standard deviation |
| --- | ---: | ---: |
| DreamBooth | 0.7427 | 0.1600 |
| DreamBooth-LoRA | 0.7002 | 0.1371 |
| Textual Inversion | 0.6541 | 0.1912 |
| IP-Adapter Plus | 0.5940 | 0.0444 |

DreamBooth has the highest measured mean DINOv2 fidelity in this pilot. Because these are whole-image embeddings, background and composition similarity can influence the values; the metric is not treated as a direct overfitting detector.

## 3. Prompt alignment

CLIP image-text similarity to the clean evaluation prompts:

| Method | Mean | Standard deviation |
| --- | ---: | ---: |
| DreamBooth-LoRA | 0.2641 | 0.0302 |
| DreamBooth | 0.2568 | 0.0322 |
| Textual Inversion | 0.2545 | 0.0276 |
| IP-Adapter Plus | 0.2066 | 0.0416 |

DreamBooth-LoRA has the highest measured mean CLIP alignment in this pilot. The difference from DreamBooth and Textual Inversion is modest and should not be generalized beyond this subject.

## 4. Qualitative observations

The predetermined seed-42 comparison grid is the main qualitative figure; it was not selected for appearance. A separate failure-case grid contains the two lowest combined DINO-plus-CLIP samples per method and is explicitly metric-selected. These sheets support visual inspection but do not replace human ratings.

## 5. Overfitting / background leakage

The 96-row manual annotation template has been prepared with blank fields for background leakage, pose copying, and composition copying. No subjective labels were assigned automatically. The inspection figure places the same predetermined seed and prompt conditions across methods for human review. Consequently, this pilot does not yet make a scored claim about which method copies backgrounds or poses most strongly.

## 6. Computational cost

| Method | Training time | Peak training VRAM | Trainable parameters | Per-subject storage | Inference latency | Peak inference VRAM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DreamBooth | 10m 49s | 18.16 GiB | 859,520,964 | 5000.109 MiB | 2.792 ± 0.110 s | 2.64 GiB |
| DreamBooth-LoRA | 5m 08s | 7.82 GiB | 3,188,736 | 12.200 MiB | 2.900 ± 0.005 s | 2.64 GiB |
| Textual Inversion | 54m 11s | 4.74 GiB | 768 | 0.003 MiB | 2.777 ± 0.057 s | 2.64 GiB |
| IP-Adapter Plus | 0 | N/A | 0 per subject | 0 | 2.908 ± 0.006 s | 3.92 GiB |

IP-Adapter Plus requires no subject-specific training or parameter storage, but uses a 150.71 MiB shared adapter checkpoint, a separately shared image encoder, and one reference image during inference. It is therefore not cost-free. DreamBooth stores a complete personalized pipeline, LoRA stores low-rank weights, and Textual Inversion stores one learned 768-dimensional embedding.

## 7. Limitations of the pilot

This is a single-subject pilot and should not be interpreted as the final cross-dataset conclusion. DINOv2 fidelity is based on whole images, manual overfitting scores remain pending human annotation, and the reference-image requirement reflects the tested configuration rather than a universal minimum. IP-Adapter model pretraining cost is outside the per-subject benchmark.
