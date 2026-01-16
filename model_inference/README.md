## 🚀 Quick Start:

Follow the instruction to run model inference.

---

### **Step 1: Prepare Data**  
Follow the provided JSON file as an example to build your own inference dataset.

The provided JSON files contain different prompts for each function described in the paper.

| Installation Script       | Input Required | Function                              |
|---------------------------|----------------|------------------------------------------------|
| `examine_chest_xray.json`   | Image Path     | Generate descriptions of an chest X-ray image by mimicking radiologist's examination process                     |
| `attention_extraction.json` | Image Path, Disease Description | Generate attention areas of a diseae description for an chest X-ray image    |



#### **Step 2: Run Inference**  
```bash
CUDA_VISIBLE_DEVICES=0 python run_inference.py --model_name_or_path --dataset --template qwen2_vl
```
Input the model path and file name of the dataset json
