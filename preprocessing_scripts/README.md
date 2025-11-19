## Data Processing Pipeline

Follow these steps in order to prepare the full dataset and generate training data.

### 1. Clone the Repository and Prepare Raw Data (Required)
```bash
git clone https://github.com/KennyLee123/GazeX.git
cd GazeX
```

Download the following datasets from PhysioNet (requires credentialed access):
- [REFLACX](https://physionet.org/content/reflacx-xray-localization/1.0.0/)
- [MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.1.0/)

Place all downloaded files in the current directory.

### 2. Generate Phase-Specific Eye-Tracking Attention Videos
```bash
python Generate_Attention_Videos_Phase_1.py
python Generate_Attention_Videos_Phase_2.py
python Generate_Attention_Videos_Phase_3.py
```

This creates:
- Attention videos stored in `./data`
- A metadata CSV (`patch_metadata_enhanced_1.csv`) with the following columns:
  
| Column             | Description                                      |
|--------------------|--------------------------------------------------|
| id                 | Unique examination ID                            |
| gaze_csv           | Path to raw gaze data                            |
| image_path         | Path to the corresponding chest X-ray           |
| patch_sentences    | Radiological findings/descriptions for each fixation patch |
| patch_intervals    | Time intervals (in seconds) for each described patch |
| frame_index        | List of frame indices corresponding to each description |

### 3. Generate Bounding Boxes and Attention Centroids per Phase
```bash
python Bounding_Box_Centroid_Phase_1.py
python Bounding_Box_Centroid_Phase_2.py
python Bounding_Box_Centroid_Phase_3.py
```

Outputs:
- Bounding box sequences and centroid coordinates for each fixation
- A CSV (`patch_metadata_enhanced_bbox_1.csv`) containing:
  - `patch_sentences`: Radiological description
  - `patch_intervals`: Time interval of the fixation
  - `bboxes_seq`: List of bounding boxes and centroids [x_min, y_max, x_max, y_max, x_centroid, y_centroid] over time

### 4. Extract Disease-Specific Attention Videos
```bash
python Extract_Disease_Based_Attention_Videos.py
```

Generates clipped video segments focusing only on fixations related to specific diseases/findings (pooled across all radiologists).

### 5. Create Shuffled Videos for Training
```bash
python Shuffle_Videos_Disease_Based.py   # Disease-focused shuffled sequences
python Shuffle_Videos_Global.py          # Globally shuffled sequences (negative/control)
```

These scripts produce temporally shuffled versions of the attention videos used as negative samples.

### 6. Generate Final Training Dataset
```bash
python Create_Training_Data.py
```

This final script:
- Organizes data for different pretraining tasks
- Generates all required prompts
- Saves everything in a model-ready format (e.g., JSONL)

