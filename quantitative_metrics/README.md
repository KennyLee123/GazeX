### **Calculate TP,FN and FP of model inference results and radiologists' examination results**  
1. Prepare inference results from model and radiologists' examination results by following the provided JSON format.

2. Run:
```bash
python compare_gazex_radiologists_examination.py
```


#### **Step 2: Run Inference**  
```bash
CUDA_VISIBLE_DEVICES=0 python run_inference.py --model_name_or_path --dataset --template qwen2_vl
```
Input the model path and file name of the dataset json

