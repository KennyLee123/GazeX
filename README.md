## Seeing Through Experts' Eyes: A Foundational Vision-Language Model Trained on Radiologists’ Gaze and Reasoning
## 📁 Directory Structure

```
.
├── preprocessing_scripts   # Scripts for preprocess required data for model training
├── data                    # Data files 
├── environments            # Reproducible Python environments  
├── model_inference         # Model and its inference scripts
├── quantitative_metrics    # Scripts for evaluating models' performance
├── LICENSE  
└── README.md  
```

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/KennyLee123/GazeX.git
cd GazeX
```

### 2. Set Up for Training and Inferencing GazeX

Python along with required packages, are summarized in the **`environments/`** directory.  
Use the provided installation scripts to set up corresponding environments.

| Installation Script       | Python Version |
|---------------------------|----------------------|
| `gazex_env_installation.sh`    | Python 3.8      |

To install the required environment for a specific setup, run:

```bash
bash environments/gazex_env_installation.sh
```
