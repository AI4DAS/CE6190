# Results

This file is generated from measured outputs by `scripts/evaluate_all.sh`. The current stage contains only the one-subject IP-Adapter Plus pilot, so it cannot support cross-method rankings.

## Measured pilot

- IP-Adapter Plus, DreamBooth `dog2`, scale 0.7, 24 outputs: DINOv2 mean-reference similarity **0.5940 ± 0.0444**; CLIP prompt similarity **0.1993 ± 0.0402**.
- Mean 50-step inference latency after warm-up was approximately **2.9 seconds per image** on one H200 NVL 1g.35gb MIG slice.
- The predetermined seed-42 grid shows strong reference-composition/background retention and weak response to requested beach, snow, kitchen, red-hat, and watercolor changes. This is a qualitative pilot observation requiring manual annotation; DINO/CLIP scores are not treated as leakage detectors.

No claims are yet made about the highest fidelity, alignment, cost, checkpoint size, or overfitting across methods. Those require the three prepared training jobs and the same standardized generation/evaluation pipeline.

