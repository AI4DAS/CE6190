# Modern Extension Smoke-Test Report

Test date: 2026-10-12

These are compatibility tests, not the 144-image modern benchmark. No existing
classical output, metric, or figure was modified.

## Results

| Test | Status | Resolution | Steps | Load time | Generation latency | Peak allocated VRAM |
|---|---|---:|---:|---:|---:|---:|
| DreamO v1.1 bf16/offload | FAIL during load | — | — | — | — | 31.24 GiB allocated before failure |
| DreamO v1.1 int8/offload | PASS | 768×768 | 12 | 264.41 s | 28.88 s | 16.04 GiB |
| UNO fp8/offload, one reference | PASS | 512×512 | 25 | 312.93 s | 39.60 s | 13.45 GiB |
| UNO fp8/offload, two references | PASS | 512×512 | 25 | shared | 44.08 s | 13.42 GiB |

## DreamO bf16 failure

The official bf16 path failed at `pipeline.to(cuda)` before generation. The
DreamO process held 31.39 GiB and another user's process on that MIG held about
1.05 GiB, leaving only 26.94 MiB. The next 36 MiB allocation raised
`torch.OutOfMemoryError`. The process exited and released its memory.

The retry used the official `quant=int8` plus CPU-offload path. It preserved
DreamO v1.1, the FLUX.1-dev backbone, Turbo 12-step sampling, reference image,
prompt, and seed. It succeeded without source-level changes to DreamO.

## Outputs

- `outputs/modern_extension/smoke/dreamo_dog2.png`
- `outputs/modern_extension/smoke/dreamo_dog2.json`
- `outputs/modern_extension/smoke/uno_dog2_single.png`
- `outputs/modern_extension/smoke/uno_dog2_single.json`
- `outputs/modern_extension/smoke/uno_dog2_multi.png`
- `outputs/modern_extension/smoke/uno_dog2_multi.json`

All three PNG outputs opened successfully with Pillow and each has a matching
metadata JSON record.

## Checkpoint storage now present

- FLUX.1-dev Hub cache: approximately 54 GB
- UNO T5 encoder cache: approximately 8.9 GB
- UNO LoRA cache: approximately 1.8 GB
- DreamO local auxiliary weights: approximately 3.8 GB

## Full-run planning estimate

For the required 72 standard images per model, measured steady-state latency
projects to approximately 35 minutes for DreamO plus its 4.4-minute load and
approximately 48 minutes for UNO plus its 5.2-minute load. Sequential runtime
is therefore roughly 1.5 hours before metadata checks and evaluation. This is
an estimate; prompt and reference complexity may change latency.

The main UNO comparison should use one reference to match the declared
72-image/model total and to keep its input requirement explicit. The verified
multi-reference capability should be evaluated as a separately labeled
extension subset rather than silently mixed into the main UNO aggregate.

## Process cleanup

Both successful smoke-test processes and the failed bf16 process exited. No
DreamO or UNO process was left running after validation.

## Prepared full-run commands (not launched)

DreamO, 72 images:

```bash
cd /home/zwu/project/CE6190/personalization-benchmark
source scripts/mig_devices.sh
CUDA_VISIBLE_DEVICES="$MIG1" \
  bash scripts/jobs/generate_dreamo_extension.sh \
  2>&1 | tee logs/modern_extension/dreamo_full.log
```

UNO single-reference, 72 images:

```bash
cd /home/zwu/project/CE6190/personalization-benchmark
source scripts/mig_devices.sh
CUDA_VISIBLE_DEVICES="$MIG2" \
  bash scripts/jobs/generate_uno_extension.sh \
  2>&1 | tee logs/modern_extension/uno_full.log
```

Both job scripts refuse to reuse their final output directory and therefore
cannot silently overwrite a previous extension run. They passed Bash syntax,
Python bytecode, CLI-help, and 3×6×4=72 configuration-count checks.
