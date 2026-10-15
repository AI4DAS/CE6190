# Main experiment training schedule proposal

## Approval status

The eight-subject batch and the verified DreamBooth source for
`colorful_sneaker` are approved. DreamO and UNO remain outside this schedule as
separate, inference-only modern extensions after the controlled benchmark.

The eight-subject matrix contains 24 trained subject-method artifacts. Three
`dog2` artifacts already exist and must be reused, leaving 21 new jobs.

## Validated settings to preserve

| Method | Steps | Resolution | Precision | Key settings | Measured dog2 runtime | Measured peak VRAM |
|---|---:|---:|---|---|---:|---:|
| DreamBooth | 800 | 512 | FP16 | prior preservation, batch 1, gradient checkpointing, LR 5e-6, no text-encoder training | 649 s (10m49s) | 18.1641 GiB |
| DreamBooth-LoRA | 800 | 512 | FP16 | rank 16, prior preservation, batch 1, gradient checkpointing, LR 1e-4 | 308 s (5m08s) | 7.8242 GiB |
| Textual Inversion | 3000 | 512 | FP16 | batch 1, gradient accumulation 4, LR 5e-4 | 3251 s (54m11s) | 4.7422 GiB |

Every process must see exactly one MIG instance as `cuda:0`. Distributed
training is prohibited.

## Live MIG inventory

The UUIDs below were reverified with `nvidia-smi -L` during the audit.

| Alias | MIG UUID | Proposed role |
|---|---|---|
| MIG0 | `MIG-eb10c6cf-0474-58a4-8e0f-7afffc48b4b3` | `cat` lane |
| MIG1 | `MIG-3dc20912-6ea3-5b44-81e4-98481ef4c763` | `pet_dog1` lane |
| MIG2 | `MIG-b0e5ab9b-29b6-5fa3-9664-7eb91cf3ef84` | `bear_plushie` lane |
| MIG3 | `MIG-9d50c02d-6036-5ad7-8817-302b8d26d7ef` | `robot_toy` lane |
| MIG4 | `MIG-8dceb4a0-261d-5b2f-9bb9-ef41f9f65765` | `teapot` lane |
| MIG5 | `MIG-a4f5c260-0bbc-52cb-b5b9-98c4b9ac3251` | `colorful_sneaker` lane |
| MIG6 | `MIG-b02034a8-d6ef-57d6-a42c-305979b1bc94` | `luggage_backpack1` lane |
| MIG7 | `MIG-8c70a6f7-6e84-54fc-a316-08ee1d2e57f7` | reserve, preparation checks, later evaluation |

## Job tracker

Pending rows intentionally have no start or end timestamp. Those fields must be
written from actual run metadata, never estimated timestamps. Estimated runtime
is shown only in the final column.

| Subject | Method | MIG UUID | Start time | End time | Status | Runtime |
|---|---|---|---|---|---|---:|
| dog2 | DreamBooth | `MIG-eb10c6cf-0474-58a4-8e0f-7afffc48b4b3` | 2026-10-10T03:27:29-07:00 | 2026-10-10T03:38:24-07:00 | COMPLETE — reuse | 649 s measured |
| dog2 | DreamBooth-LoRA | `MIG-3dc20912-6ea3-5b44-81e4-98481ef4c763` | 2026-10-10T03:38:43-07:00 | 2026-10-10T03:43:59-07:00 | COMPLETE — reuse | 308 s measured |
| dog2 | Textual Inversion | `MIG-b0e5ab9b-29b6-5fa3-9664-7eb91cf3ef84` | 2026-10-10T03:05:50-07:00 | 2026-10-10T04:00:08-07:00 | COMPLETE — reuse | 3251 s measured |
| cat | Textual Inversion | `MIG-eb10c6cf-0474-58a4-8e0f-7afffc48b4b3` | 2026-10-10T07:49:26-07:00 | — | RUNNING | accumulating |
| cat | DreamBooth | `MIG-eb10c6cf-0474-58a4-8e0f-7afffc48b4b3` | — | — | QUEUED | ~649 s |
| cat | DreamBooth-LoRA | `MIG-eb10c6cf-0474-58a4-8e0f-7afffc48b4b3` | — | — | QUEUED | ~308 s |
| pet_dog1 | Textual Inversion | `MIG-3dc20912-6ea3-5b44-81e4-98481ef4c763` | 2026-10-10T07:49:26-07:00 | — | RUNNING | accumulating |
| pet_dog1 | DreamBooth | `MIG-3dc20912-6ea3-5b44-81e4-98481ef4c763` | — | — | QUEUED | ~649 s |
| pet_dog1 | DreamBooth-LoRA | `MIG-3dc20912-6ea3-5b44-81e4-98481ef4c763` | — | — | QUEUED | ~308 s |
| bear_plushie | Textual Inversion | `MIG-b0e5ab9b-29b6-5fa3-9664-7eb91cf3ef84` | 2026-10-10T07:49:27-07:00 | — | RUNNING | accumulating |
| bear_plushie | DreamBooth | `MIG-b0e5ab9b-29b6-5fa3-9664-7eb91cf3ef84` | — | — | QUEUED | ~649 s |
| bear_plushie | DreamBooth-LoRA | `MIG-b0e5ab9b-29b6-5fa3-9664-7eb91cf3ef84` | — | — | QUEUED | ~308 s |
| robot_toy | Textual Inversion | `MIG-9d50c02d-6036-5ad7-8817-302b8d26d7ef` | 2026-10-10T07:49:27-07:00 | — | RUNNING | accumulating |
| robot_toy | DreamBooth | `MIG-9d50c02d-6036-5ad7-8817-302b8d26d7ef` | — | — | QUEUED | ~649 s |
| robot_toy | DreamBooth-LoRA | `MIG-9d50c02d-6036-5ad7-8817-302b8d26d7ef` | — | — | QUEUED | ~308 s |
| teapot | Textual Inversion | `MIG-8dceb4a0-261d-5b2f-9bb9-ef41f9f65765` | 2026-10-10T07:49:24-07:00 | — | RUNNING | accumulating |
| teapot | DreamBooth | `MIG-8dceb4a0-261d-5b2f-9bb9-ef41f9f65765` | — | — | QUEUED | ~649 s |
| teapot | DreamBooth-LoRA | `MIG-8dceb4a0-261d-5b2f-9bb9-ef41f9f65765` | — | — | QUEUED | ~308 s |
| colorful_sneaker | Textual Inversion | `MIG-a4f5c260-0bbc-52cb-b5b9-98c4b9ac3251` | 2026-10-10T07:49:24-07:00 | — | RUNNING | accumulating |
| colorful_sneaker | DreamBooth | `MIG-a4f5c260-0bbc-52cb-b5b9-98c4b9ac3251` | — | — | QUEUED | ~649 s |
| colorful_sneaker | DreamBooth-LoRA | `MIG-a4f5c260-0bbc-52cb-b5b9-98c4b9ac3251` | — | — | QUEUED | ~308 s |
| luggage_backpack1 | Textual Inversion | `MIG-b02034a8-d6ef-57d6-a42c-305979b1bc94` | 2026-10-10T07:49:27-07:00 | — | RUNNING | accumulating |
| luggage_backpack1 | DreamBooth | `MIG-b02034a8-d6ef-57d6-a42c-305979b1bc94` | — | — | QUEUED | ~649 s |
| luggage_backpack1 | DreamBooth-LoRA | `MIG-b02034a8-d6ef-57d6-a42c-305979b1bc94` | — | — | QUEUED | ~308 s |

Within each subject lane, jobs are serialized so no two processes share a MIG.
Textual Inversion is listed first because it dominates runtime; DreamBooth and
LoRA then complete on the same lane. This is experiment-level parallelism only.

## Class-image preparation proposal

After approval, DreamBooth and LoRA may share one image-only prior directory for
each semantic class. Metadata must be stored beside, never inside, that image
directory. The existing 200-image dog set remains available for `dog2` and must
not be regenerated.

Proposed new class prompts are `cat`, `stuffed animal`, `toy`, `teapot`,
`sneaker`, and `backpack`: six new 200-image sets. `pet_dog1` reuses the
validated dog class set. Their exact
prompts and directories must be reviewed before generation. Class images are
training auxiliaries and are forbidden as DINOv2 references, CLIP references,
or qualitative references.

## Runtime estimate

Measured dog2 times are used directly; no throughput scaling is assumed.

- One subject lane: about 70m08s for TI + DreamBooth + LoRA.
- Seven pending lanes: 8.18 aggregate training GPU-hours.
- Complete eight-subject matrix, including already-spent dog2 time: 9.35
  aggregate training GPU-hours.
- Ideal pending wall time on seven simultaneous MIG lanes: about 1h10m.
- Recommended planning allowance for validation, I/O, and subject variation:
  1.5–2.0 hours wall time and 10.2–12.3 pending training GPU-hours.
- Seven new class-image sets: budget approximately 1–2 additional GPU-hours
  until measured; they can be prepared independently after approval.
- The 672 new classical evaluation images should require roughly 0.5–0.7
  aggregate inference GPU-hours at the measured pilot latency, excluding model
  loading and metric evaluation.

These estimates do not include DreamO/UNO, which are explicitly deferred.

## Storage estimate

Actual retained dog2 directories, including all resumable checkpoints, occupy:

- DreamBooth: 43.309 GiB;
- LoRA: 0.159 GiB;
- Textual Inversion: 4.458 GiB;
- total retained per subject: 47.926 GiB.

If the same checkpoint retention policy is used:

- seven new subjects require approximately 335.48 GiB;
- all eight subjects require approximately 383.41 GiB, including existing dog2;
- six new class-image sets add about 0.12 GiB at the measured dog-class size;
- 672 new evaluation images add about 0.24 GiB at the pilot compression rate.

Reserve at least **400 GiB incremental space** before launching the pending
queue to allow subject-dependent image sizes, logs, validation images, and
temporary files. The audited filesystem currently has about 5.0 TiB free.

If intermediate checkpoints are later archived or removed under a separate,
explicit retention decision, final personalized artifacts alone are estimated
at about 34.3 GiB for the seven new subjects and 39.2 GiB for all eight. No
existing checkpoint is deleted or altered by this proposal.
