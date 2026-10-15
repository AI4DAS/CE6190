# Main experiment: eight-subject selection audit

## Status and scope

This audit records the approved eight-subject first batch. It does not modify
the completed `dog2` pilot. No
subject preprocessing, class-image generation, model training, model inference,
or DreamO/UNO work was performed.

The repository, current configurations, completed checkpoints, reference-image
directories, live MIG inventory, and pilot results were inspected before this
audit was written. All selected reference images were opened successfully with
Pillow and visually reviewed.

## Required source correction

The requested list assigns `colorful_sneaker` to CustomConcept101, but that
subject does **not** exist in the local CustomConcept101 benchmark. It exists as
an official Google DreamBooth subject at:

`data/dreambooth/upstream/dataset/colorful_sneaker/`

This audit uses that real DreamBooth source and labels it explicitly; it does not
silently fabricate or relabel a CustomConcept101 subject. The source correction
was approved. The final batch contains six DreamBooth subjects and two
CustomConcept101 subjects.

## Audited subject table

Resolution entries describe original references. A superscript-like `²` means a
square image, for example `1280²` means `1280×1280`.

| Subject | Verified dataset | Category | References | Original resolution | Difficulty | Reason for inclusion |
|---|---|---|---:|---|---|---|
| `dog2` | DreamBooth | Animal | 6 | 1274²–1441² | Medium | Completed pilot baseline; tests dense fur, facial proportions, pose variation, and continuity with validated results |
| `cat` | DreamBooth | Animal | 5 | 1277²–1282² | Hard | Cross-species identity test with subtle tabby markings and correlated outdoor backgrounds that may expose leakage |
| `pet_dog1` | CustomConcept101 | Animal | 8 | 956²–1298²; one 965×968 | Medium | Same-category variation with a gray short-haired dog, pale eyes, long ears, varied poses, and diverse backgrounds |
| `bear_plushie` | DreamBooth | Soft texture object | 5 | 2048² | Easy | Strong color blocks and crown provide an easier identity baseline while fuzzy material tests soft texture preservation |
| `robot_toy` | DreamBooth | Rigid object | 5 | 1440² | Hard | Articulated body, bilateral structure, wheels, blue trim, joints, and face display test geometry preservation |
| `teapot` | DreamBooth | Rigid object | 5 | 703²–1282² | Medium | Distinctive spout, lid, handle, dark material, lighting changes, and scale variation test shape/material consistency |
| `colorful_sneaker` | **DreamBooth** (source correction) | Wearable/product | 5 | 2048² | Hard | Pastel panel pattern, seams, sole geometry, and lacing test color and texture preservation; requested CC101 path is absent |
| `luggage_backpack1` | CustomConcept101 | Wearable/product | 5 | 1114²–2287² | Medium | Leather flap, buckles, pocket, straps, and repeated cropped viewpoints test complex product structure |

## Distribution

| Category | Count | Subjects |
|---|---:|---|
| Animal | 3 | `dog2`, `cat`, `pet_dog1` |
| Soft texture object | 1 | `bear_plushie` |
| Rigid object | 2 | `robot_toy`, `teapot` |
| Wearable/product | 2 | `colorful_sneaker`, `luggage_backpack1` |
| **Total** | **8** | |

| Difficulty | Count |
|---|---:|
| Easy | 1 |
| Medium | 4 |
| Hard | 3 |

| Verified dataset | Count |
|---|---:|
| DreamBooth | 6 |
| CustomConcept101 | 2 |

## Reference integrity and privacy

- Every selected subject exists at its verified local path.
- Reference counts range from 5 to 7 images. All files are non-empty, use
  supported JPEG or PNG formats, and decode successfully.
- `dog2` retains its original six public DreamBooth references. The existing
  processed pilot copy and all completed checkpoints remain untouched.
- No selected subject is a person, and no selected image contains an identifiable
  face. Some public product photos show cropped arms or a cropped torso; these
  regions are incidental and the personalization target is exclusively the
  object.
- No private images were added. Dataset source, attribution, and published
  license notes remain documented in `results/dataset_documentation.md` and the
  preserved upstream repositories.
- Prior-preservation class images were not considered reference images and are
  excluded from all contact sheets and future evaluation.

## Contact sheets

One reference-only contact sheet was created per subject:

- `results/figures/main_subject_selection/dog2_contact_sheet.png`
- `results/figures/main_subject_selection/cat_contact_sheet.png`
- `results/figures/main_subject_selection/pet_dog1_contact_sheet.png`
- `results/figures/main_subject_selection/bear_plushie_contact_sheet.png`
- `results/figures/main_subject_selection/robot_toy_contact_sheet.png`
- `results/figures/main_subject_selection/teapot_contact_sheet.png`
- `results/figures/main_subject_selection/colorful_sneaker_contact_sheet.png`
- `results/figures/main_subject_selection/luggage_backpack1_contact_sheet.png`

The category overview is:

`results/figures/main_subject_selection/all_subjects_overview_approved8.png`

These figures contain only existing public reference images. No generated image
or model inference was used. The reproducible renderer is
`src/visualization/main_subject_selection.py` and refuses to overwrite existing
figures unless `--overwrite` is explicitly supplied.

## Existing dog2 artifacts and job-count correction

The three `dog2` training artifacts are already complete and validated:

| Method | Existing output | Status |
|---|---|---|
| DreamBooth | `outputs/training/dreambooth/dog2_db_s800_seed42` | SUCCESS |
| DreamBooth-LoRA | `outputs/training/lora/dog2_lora_r16_s800_seed42` | SUCCESS |
| Textual Inversion | `outputs/training/textual_inversion/dog2_ti_s3000_seed42` | SUCCESS |

The final eight-subject matrix contains 24 trained subject-method artifacts, but
only **21 new training jobs** remain: seven new subjects multiplied by three
training methods. Retraining `dog2` would violate the stated constraints and is
not included in the proposed queue.

## Approval status

The eight-subject batch, the verified DreamBooth source for
`colorful_sneaker`, and the 21-new-job schedule were approved. Storage policy
was tightened before launch: no new method retains intermediate training
checkpoints. Existing dog2 artifacts remain unchanged.
