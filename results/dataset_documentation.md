# Dataset documentation

## Sources

- Google DreamBooth dataset: official Google DreamBooth repository, <https://github.com/google/dreambooth>. Its `references_and_licenses.txt` attributes author-captured and Unsplash-sourced images individually. The selected `dog2` images are attributed upstream to Unsplash photographer Lukasz Rawa; the upstream license/reference file is preserved in the clone.
- CustomConcept101: official Adobe Custom Diffusion repository, <https://github.com/adobe-research/custom-diffusion>. Its dataset README states that Unsplash images use the Unsplash License, creator-captured images are released under CC BY-SA 4.0, and flower-source URLs are supplied for attribution. The repository’s Adobe Research License applies to specified repository materials; this summary does not reinterpret it as the image license.

## Pilot selection

The pilot uses the non-human `dog2` subject from the Google DreamBooth dataset: 6 images. Image dimensions, upstream license note, and source paths are generated into `results/metrics/dataset_summary.csv` by the preparation script.

Identifiable human subjects and faces are excluded from the main benchmark. No private photographs are used.

## Ethics and privacy statement

Our experiments use publicly available academic datasets and focus primarily on non-human subjects such as animals and objects to reduce identity and privacy concerns. We preserve dataset citations and licensing information. Although the evaluated personalization methods may also be applied to identifiable individuals, such applications raise additional concerns related to consent, impersonation, and misuse.

This document reports upstream license statements and does not offer a legal interpretation.

