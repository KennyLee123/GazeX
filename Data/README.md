## Datasets

Currently included datasets:

- [REFLACX](https://physionet.org/content/reflacx-xray-localization/1.0.0/)
- [MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.1.0/)

### REFLACX
REFLACX provides eye-tracking data collected while radiologists dictated reports for frontal chest x-rays from the MIMIC-CXR dataset, paired with the timestamped transcription of the dictation.
### MIMIC-CXR
The MIMIC Chest X-ray (MIMIC-CXR) Database is a large publicly available dataset of chest radiographs in DICOM format with free-text radiology reports. The dataset contains 377,110 images corresponding to 227,835 radiographic studies performed at the Beth Israel Deaconess Medical Center in Boston, MA.
---
## Steps to download and organize data

1. Register PhysioNet account with credentialed access to dataset REFLACX and MIMIC-CXR.

2. Download the datasets from PhysioNet and place all downloaded files in the directory **`preprocessing_scripts/`**.

3. Follow the instruction in **`preprocessing_scripts/README.md`** to process all data for training and inferencing.
