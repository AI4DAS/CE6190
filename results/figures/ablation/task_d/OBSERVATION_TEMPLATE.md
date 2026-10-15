# Task D visual inspection

This is a non-blind visual inspection of seed 42 only, so it does not establish
a general cause.

- DreamBooth, cat and dog2: the conspicuous high-frequency face/body blotches
  visible in the default FP16 row are absent or strongly reduced in the CFG 5.0
  row. The FP32 row remains visually close to the default FP16 row and retains
  the same conspicuous blotches in the affected prompt slots.
- LoRA, cat and dog2: CFG 5.0 and adapter weights 0.8/0.6 reduce or remove the
  most conspicuous blotches visible at CFG 7.5 / weight 1.0. Weight 0.6 also
  visibly changes subject identity in some dog2 outputs, so artifact reduction
  is not cost-free. FP32 alone does not consistently remove the blotches.

These images support an inference-setting sensitivity observation; they do not
identify the training-time cause of the artifacts.
