## 🚀 Quick Start:

Follow the instruction to run model inference.

---

### **Step 1: Prepare Data**  
Follow the provided JSON file as an example to build your own inference dataset.

The provided JSON files contain different prompts for each function described in the paper.

| Installation Script       | Python/Julia Version | Methods Supported                              |
|---------------------------|----------------------|------------------------------------------------|
| `env1_installation.sh`    | Python 3.8           | UCE, SCimilarity                               |
| `env2_installation.sh`    | Python 3.11          | Geneformer, scFoundation, scVI, PCA, LDVAE     |
| `env3_installation.sh`    | Python 3.10          | scGPT                                          |


#### **Step 2: Run Inference**  
```bash
CUDA_VISIBLE_DEVICES=0 python run_inference.py --model_name_or_path --dataset --template qwen2_vl
```
Input the model path and file name of the dataset json
