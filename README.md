# 3CE_team

---

## 🔧 Installation

1. Clone the repository  

       git clone -b feature/integrate-teammate-models https://github.com/ChunYi-Lin/dlcv-fall-2025-final
       cd dlcv-fall-2025-final

2. Create and activate a conda environment with Python 3.10

       conda create -n spatialagent python=3.10 -y
       conda activate spatialagent

3. Install Python dependencies (Adjust pytorch installation with your CUDA version)

       pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu118
       pip install -r requirements.txt
       pip install autoawq
       pip install vllm
---

## 📦 Preparation

1. Model checkpoints be downloaded using:

       gdown -O best_acc_model.pth 1q6aFUsTs6BYcYsxuiWtBRUJzVOQ6Jl40
       gdown -O best_loss_model.pth 1pgZqm9J9oV_hbd46SqhhW3XKHMCjlWPr
       gdown -O epoch_4.pth 1TExckowE8RjfTsEzV5uzLmJx6_q-NwiC

2. Place the downloaded files in corresponding directory following the below Project Structure.

3. Download the [DLCV_Final1](https://huggingface.co/datasets/yaguchi27/DLCV_Final1) and put in data dir following project structure.

---

## 📂 Project Structure

    dlcv-fall-2025-final/
    ├── agent/
    │   └── agent_run.py
    ├── distance_est/
    │   └── ckpt/
    │       ├── best_acc_model.pth
    │       └── best_loss_model.pth
    ├── inside_pred/
    │   └── ckpt/
    │       └── epoch_4.pth
    ├── utils/
    ├── data/
    │   ├── train/
    │   │   ├── images/
    │   │   ├── depths/
    │   │   └── train.json
    │   ├── val/
    │   │   ├── images/
    │   │   ├── depths/
    │   │   └── val.json
    │   └── test/
    │       ├── images/
    │       ├── depths/
    │       └── test.json
    ├── output/
    ├── Qwen2.5-72B-Instruct-AWQ/
    ├── download_model.py
    ├── requirements.txt
    └── README.md

---

## 🧠 Usage

### 1. Inference on test set

```bash
python3 agent/agent_run.py --split test \
               --model ./Qwen2.5-72B-Instruct-AWQ \
               --output_path ./output/test.json \
               --max_new_tokens 4096 \
               --do_sample \
               --temperature 0.2 \
               --llm_backend vllm
```

You can swap the LLM with any Hugging Face Transformers causal LM (local path or hub id):

```bash
cd agent
python3 agent_run.py --output_path ../output/test.json --model <model_name_or_path> --device_map auto --dtype bf16
```

## ⚒️ QA Data Pre-processing and Model Training (Optional)

### 0. QA Data Pre-processing

To pre-process the QA, you need to update the below script with your Google API key.
Note that this step is optional because data.zip already provide the processed QA data.

```
python3 utils/question_rephrase.py --split test --model ./Qwen2.5-72B-Instruct-AWQ/ --llm_backend vllm
```


We provide the pre-trained model checkpoint, but we also provide the training script of our model as follows.

### 1. Train the inclusion classification model

```
cd utils
python3 organize_inside.py
cd ..
cd inside_pred
python3 train.py
```


---

## 📚 Reference

```bibtex
@InProceedings{Huang_2025_ICCV,
    author    = {Huang, Hsiang-Wei and Kim, Pyongkun and Cheng, Jen-Hao and Chen, Kuang-Ming and Yang, Cheng-Yen and Alattar, Bahaa and Lin, Yi-Ru and Kim, Sangwon and Kim, Kwangju and Huang, Chung-I and Hwang, Jenq-Neng},
    title     = {Warehouse Spatial Question Answering with LLM Agent: 1st Place Solution of the 9th AI City Challenge Track 3},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops},
    month     = {October},
    year      = {2025},
    pages     = {5224-5228}
}
```
