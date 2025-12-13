# SpatialAgent

**1st Place Solution of the ICCV 2025 AI City Challenge, Track 3.**

📄 **[ICCVW 2025 Paper](https://openaccess.thecvf.com/content/ICCV2025W/AICity/papers/Huang_Warehouse_Spatial_Question_Answering_with_LLM_Agent_1st_Place_Solution_ICCVW_2025_paper.pdf)**

<p align="center">
  <img src="asset/leaderboard.png" alt="Leaderboard Result"/>
</p>

---

## 🔧 Installation

1. Clone the repository  

       git clone https://github.com/hsiangwei0903/SpatialAgent.git
       cd SpatialAgent

2. Create and activate a conda environment with Python 3.10

       conda create -n spatialagent python=3.10 -y
       conda activate spatialagent

3. Install Python dependencies (Adjust pytorch installation with your CUDA version)

       pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu118
       pip install -r requirements.txt

---

## 📦 Preparation

1. Model checkpoints and pre-processed QA data can be downloaded from [here](<https://drive.google.com/drive/u/1/folders/1_ovPjqADpvM0fQdNBLAPdWiemC5MFaG7>).

2. Place the downloaded files in corresponding directory following the below Project Structure.

3. Download the [AI City Challenge PhysicalAI Spatial Intelligence dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-Spatial-Intelligence-Warehouse) and put in data dir following project structure.

---

## 📂 Project Structure

    SpatialAgent
    ├── agent
    ├── distance_est/
    │   └──  ckpt/
    │       ├── 3m_epoch6.pth
    │       └── epoch_5_iter_6831.pth
    ├── inside_pred/
    │   └── ckpt/
    │       └── epoch_4.pth
    ├── utils
    ├── data/
    │   ├── train
    │   ├── val
    │   └── test/
    │       └── images/
    │       └── depths/
    └── README.md

---

## 🧠 Usage

### 1. Inference on test set

```bash
cd agent
python agent_run.py --output_path ../output/test.json --quantization none
```

You can swap the LLM with any Hugging Face Transformers causal LM (local path or hub id):

```bash
cd agent
python agent_run.py --output_path ../output/test.json --model <model_name_or_path> --device_map auto --dtype bf16
```

Additionally, some QA might failed because Gemini return invalid format or answer, run again with thinking mode enabled can solve this issue. 
Running this command will re-run those failure cases.
```

cd agent
python agent_run.py --project_id <your Vertex AI API> --think_mode

```

## ⚒️ QA Data Pre-processing and Model Training (Optional)

### 0. QA Data Pre-processing

To pre-process the QA, you need to update the below script with your Google API key.
Note that this step is optional because data.zip already provide the processed QA data.

```
python utils/question_rephrase.py
```


We provide the pre-trained model checkpoint, but we also provide the training script of our model as follows.

### 1. Train the distance estimation model

```
cd distance_est
python train.py
```

### 2. Train the inclusion classification model

```
cd inside_pred
python train.py
```


---

## 📚 Citation

If you find this work useful, please cite our ICCV Workshop 2025 paper, thank you!

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
