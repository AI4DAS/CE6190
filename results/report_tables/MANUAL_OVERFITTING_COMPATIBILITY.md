# Manual-overfitting table compatibility

`main_results_with_manual_overfitting.csv`, `manual_overfitting.csv`, and the
tables under `manual_overfitting/` were derived from the original four-method
annotation set. They must not be used as the current five-method manual result:

- their IP-Adapter ratings refer to replaced scale-0.7 images;
- the current scale-0.3 IP-Adapter images are not rated;
- the non-personalized base-model control is not rated;
- the LoRA rank/step ablation images are not rated.

The current automatic DINOv2 and CLIP tables remain valid. Fresh annotation
templates are available for the main benchmark and LoRA ablation.
