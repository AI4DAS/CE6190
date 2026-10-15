# CE6190 Storage and Upload Audit

Audit date: 2026-10-13

## Current size

- `/home/zwu/project/CE6190`: 298 GiB (`319,535,310,322` logical bytes)
- Almost all content is in `personalization-benchmark/`.
- Free local disk space at audit time: approximately 4.6 TiB.

Largest top-level components:

| Component | Size |
|---|---:|
| `outputs/` | 233 GiB |
| `outputs/training/` | 181 GiB |
| `models/` | 53 GiB |
| `data/` | 6.8 GiB |
| `third_party/` | 4.0 GiB |
| `results/` | 1.7 GiB |
| `logs/` | 42 MiB |
| `.git/` | 19 MiB |

The largest avoidable storage category is Diffusers `checkpoint-*` directories:

- All `checkpoint-*` directories: approximately 162 GiB
- `optimizer.bin` files alone: approximately 98 GiB
- Final DreamBooth models without intermediate checkpoint directories: approximately 40 GiB
- Final LoRA outputs without intermediate checkpoint directories: approximately 126 MiB
- Final Textual Inversion outputs without intermediate checkpoint directories: approximately 104 MiB

The folder contains 18,547 files, including 221 files larger than 100 MiB and
71 files larger than 1 GiB. Most bytes are already-compressed or high-entropy
model weights: about 163 GiB of `.safetensors`, 111 GiB of `.bin`, and 12 GiB
of `.ckpt`. Therefore archive compression is expected to save relatively little;
the final archive size must be measured after creation rather than assumed.

## Recommended publication split

### 1. GitHub core repository (recommended)

Publish source code, configurations, environment specifications, metric CSVs,
report tables, Markdown documentation, and a curated set of representative
figures. The currently tracked Git working set is about 20 MiB; code/configs and
tabular/documentation deliverables add only a few MiB. A practical GitHub package
should remain in the tens to low hundreds of MiB.

Do not commit:

- `models/`
- `outputs/`
- `logs/`
- downloaded/raw datasets
- `third_party/`
- large export ZIP files
- intermediate checkpoints and optimizer states

The repository currently has no configured Git remote and Git LFS is not installed.
The working tree also contains many uncommitted and untracked experiment files, so
the exact publication manifest must be reviewed before `git add`.

GitHub currently blocks ordinary Git objects larger than 100 MiB, enforces a 2 GiB
maximum per push, recommends an on-disk repository size no larger than 10 GiB, and
recommends repositories ideally remain below 1 GiB. See:

- <https://docs.github.com/en/repositories/creating-and-managing-repositories/repository-limits>
- <https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github>

### 2. Google Drive reproducibility bundle (recommended)

A curated bundle containing code, processed subject data, results, main generated
outputs, experiment outputs, and final trained artifacts—but excluding downloaded
base models, third-party clones, debug/smoke outputs, and all `checkpoint-*`
directories—is approximately 44 GiB before compression.

This is the preferred downloadable "complete experimental result" package because
the base SD1.5/IP-Adapter checkpoints and third-party repositories can be restored
from recorded upstream identifiers, while the unique trained final artifacts and
measured outputs are preserved.

### 3. Google Drive forensic archive (optional)

An exact archive of the entire project is approximately 298 GiB before compression.
Because most bytes are model weights, a conservative upload plan should budget close
to 298 GiB of Drive capacity. Excluding only `checkpoint-*` directories reduces the
working estimate to about 136 GiB.

Google Drive supports general files up to 5 TB, subject to account storage capacity.
Google Workspace documentation also states a 750 GB per-user upload/copy limit over
24 hours for shared drives. See:

- <https://support.google.com/drive/answer/37603>
- <https://support.google.com/a/users/answer/7338880>

## Upload readiness

- No Google Drive mount was found.
- `rclone`, `gdrive`, `gh`, and `git-lfs` are not currently available in `PATH`.
- No Git remote is configured.
- A filename/content scan of the publishable code/results area found no obvious
  Hugging Face, GitHub, Google, AWS, or W&B token patterns. This is a preliminary
  scan, not a guarantee; logs and metadata should still be reviewed for usernames,
  absolute paths, and private URLs before public release.

## Recommended next action

1. Create an explicit GitHub manifest and strengthen `.gitignore` before staging.
2. Create the 44 GiB Drive bundle first, plus `SHA256SUMS` and a file manifest.
3. Split the archive into recoverable chunks (for example 20–25 GiB each) if the
   upload connection is unreliable.
4. Configure an authenticated Drive client such as `rclone`, confirm at least
   50 GiB free for the curated bundle (or at least 320 GiB for the forensic copy),
   and upload with checksum verification.
5. Add the Drive archive link and exact checksum to the GitHub README instead of
   placing weights or generated archives in Git history.

No upload or deletion was performed during this audit.
