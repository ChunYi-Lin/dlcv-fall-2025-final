# Warehouse Spatial Intelligence

**NTU Deep Learning for Computer Vision — Fall 2025 Final Project**

A tool-augmented multimodal agent for answering fine-grained spatial questions in dense warehouse scenes.

Given a natural-language question together with an RGB image, depth map, and RLE-encoded region masks, the system answers spatial reasoning questions including:

- left/right spatial relations
- metric distance estimation
- multiple-choice region grounding
- object counting

This project was developed for **Challenge 1: Warehouse Spatial Intelligence** of the NTU Deep Learning for Computer Vision (DLCV) Fall 2025 Final Project, based on Track 3 of the NVIDIA AI City Challenge at ICCV 2025.

📄 **[Final Project Poster](docs/poster.pdf)**

---

## Overview

The task requires reasoning about fine-grained spatial relationships among multiple objects and regions in warehouse-scale environments.

### Input

- RGB image
- depth map
- RLE-encoded region masks
- natural-language question

### Output

A normalized answer depending on the question type, for example:

- distance in meters
- `left` / `right`
- region ID
- object count

A language model alone is not reliable for precise geometric reasoning such as metric distance and containment. Our system therefore combines a locally hosted LLM with explicit spatial tools.

---

## System Architecture

Our final system uses a **tool-augmented LLM agent** with **Qwen2.5-72B-Instruct-AWQ** as the reasoning core.

The agent interprets a question and emits structured tool calls such as:

```text
<execute>dist(pallet_1, forklift_1)</execute>
```

The corresponding spatial tool computes the result and returns it to the LLM. The agent can perform multiple reasoning steps before producing a normalized final answer:

```text
<answer>...</answer>
```

Conceptually:

```text
RGB + Depth + Region Masks + Question
                |
                v
       Qwen2.5 Reasoning Core
                |
         structured tool call
                |
                v
            Tools API
                |
     +----------+-----------+
     |          |           |
  Distance    Inside     Rule-based
   Model      Model       Utilities
     |          |           |
     +----------+-----------+
                |
                v
        normalized answer
```

The executor uses a whitelist of supported functions and structured argument parsing rather than executing arbitrary model-generated code.

---

## Spatial Tools

### Distance Estimation

Metric distance is handled by a dedicated **Siamese RGB-D regression model** instead of relying on free-form LLM estimation.

Each object branch combines:

- an RGB-D crop encoded by a ResNet-based visual backbone
- a normalized geometry vector containing bounding-box center, width, height, and area

The representations of the two objects are combined with pairwise interaction features before predicting their distance.

The final system also applies task-specific containment logic; for example, when an object is determined to be inside a buffer region, the corresponding distance is treated as zero.

### Containment Prediction

The `inside` tool predicts whether an object lies inside a container region.

Its input is a 5-channel tensor:

```text
RGB (3 channels)
+ container mask (1 channel)
+ object mask (1 channel)
```

The model can evaluate multiple candidate objects in a batch, allowing the same tool to support both containment reasoning and counting.

### Deterministic Utilities

For spatial relations that can be computed reliably without an LLM, the system uses deterministic operations, including:

- left/right comparison using mask centroids
- nearest-object selection using batched distance inference
- region-overlap based utilities
- containment-based counting

---

## Example Reasoning Trace

For a question such as:

```text
Is the pallet closer to the buffer or the forklift?
```

the agent may perform:

```text
<execute>dist(pallet_1, buffer_1)</execute>
2.45

<execute>dist(pallet_1, forklift_1)</execute>
5.12

<answer>buffer_1</answer>
```

This separates language reasoning from numerical spatial computation.

---

## Results

| Model | Count | Distance | Left / Right | MCQ | Validation | Test |
|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5-14B | 79.25 | 90.67 | 100.00 | 88.59 | 89.63 | 82.48 |
| **Qwen2.5-72B-Instruct-AWQ** | **93.08** | **97.33** | **100.00** | **90.60** | **95.25** | **95.65** |

Our final system achieved a **95.65 test score**.

The original course CodaBench leaderboard is no longer publicly available. The reported results are preserved in the [final project poster](docs/poster.pdf).

---

## My Contributions

This was a four-person team project. My main contributions (**Chun-Yi Lin**) included:

- replacing the API-dependent LLM pipeline with locally hosted Qwen inference
- making the LLM backend configurable instead of tying the agent to a single model
- adding Hugging Face Transformers and **vLLM** inference support
- adding image caching and modifying the agent inference pipeline
- extending the question-rephrasing pipeline to support train, validation, and test splits
- adapting output handling to the course evaluation format
- integrating the team's distance-estimation and containment models into the final tool-augmented agent

---

## Repository Structure

```text
dlcv-fall-2025-final/
├── agent/
│   ├── agent_run.py
│   ├── tools.py
│   ├── llm.py
│   └── ...
├── distance_est/
│   ├── model.py
│   ├── train.py
│   └── ckpt/
├── inside_pred/
│   ├── model.py
│   ├── train.py
│   └── ckpt/
├── utils/
├── docs/
│   └── poster.pdf
├── data/
│   ├── train/
│   ├── val/
│   └── test/
├── output/
├── download_model.py
├── requirements.txt
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/ChunYi-Lin/dlcv-fall-2025-final.git
cd dlcv-fall-2025-final
```

Create a Python 3.10 environment:

```bash
conda create -n spatialagent python=3.10 -y
conda activate spatialagent
```

Install PyTorch and the required packages:

```bash
pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 \
    --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt
pip install gdown autoawq vllm
```

Adjust the PyTorch installation for your CUDA version if necessary.

---

## Model Checkpoints

Create the checkpoint directories:

```bash
mkdir -p distance_est/ckpt inside_pred/ckpt
```

Download the checkpoints used by the final system:

```bash
gdown 1pgZqm9J9oV_hbd46SqhhW3XKHMCjlWPr \
    -O distance_est/ckpt/best_loss_model.pth

gdown 1TExckowE8RjfTsEzV5uzLmJx6_q-NwiC \
    -O inside_pred/ckpt/epoch_4.pth
```

An additional distance-model checkpoint is available with:

```bash
gdown 1q6aFUsTs6BYcYsxuiWtBRUJzVOQ6Jl40 \
    -O distance_est/ckpt/best_acc_model.pth
```

---

## Dataset Preparation

Download the DLCV final-project dataset:

- [DLCV_Final1 on Hugging Face](https://huggingface.co/datasets/yaguchi27/DLCV_Final1)

Place the data under:

```text
data/
├── train/
│   ├── images/
│   ├── depths/
│   └── train.json
├── val/
│   ├── images/
│   ├── depths/
│   └── val.json
└── test/
    ├── images/
    ├── depths/
    └── test.json
```

---

## Question Pre-processing

The project includes a question-rephrasing stage used before agent inference.

For example:

```bash
python3 utils/question_rephrase.py \
    --split test \
    --model ./Qwen2.5-72B-Instruct-AWQ \
    --llm_backend vllm
```

The same script supports:

```text
--split train
--split val
--split test
```

By default, the processed data are written to the corresponding dataset directory unless an explicit output path is provided.

---

## Inference

### Final Qwen + vLLM configuration

```bash
python3 agent/agent_run.py \
    --split test \
    --model ./Qwen2.5-72B-Instruct-AWQ \
    --output_path ./output/test.json \
    --max_new_tokens 4096 \
    --do_sample \
    --temperature 0.2 \
    --llm_backend vllm
```

The agent writes normalized predictions to:

```text
output/test.json
```

and stores the generated conversations in a separate output file.

### Using another Hugging Face model

The LLM backend is configurable, so another Hugging Face causal language model can also be used:

```bash
python3 agent/agent_run.py \
    --split test \
    --model <model_name_or_path> \
    --device_map auto \
    --dtype bf16 \
    --llm_backend hf
```

---

## Training

Training code for the learned spatial components is provided under:

```text
distance_est/
inside_pred/
```

For example, the containment model can be trained with:

```bash
cd utils
python3 organize_inside.py
cd ../inside_pred
python3 train.py
```

The distance-estimation training implementation is available in `distance_est/train.py`.

Some training scripts retain experiment-specific dataset or checkpoint paths from the original course workspace and may require path adjustments in a new local environment. The checkpoints used for the reported final results are provided in the [Model Checkpoints](#model-checkpoints) section above.

---

## Course Constraints

The final-project rules prohibited external API models and required submitted results to be reproducible from the submitted code and models.

For this reason, our submitted pipeline used a locally hosted Qwen model rather than relying on a remote language-model API.

---

## Team

**3CE Team**

- Jen-Chieh Yang
- Chun-Yi Lin
- Chen-Kai Wen
- Chih-Hsuan Chen

---

## Acknowledgements

The final-project challenge is based on the Warehouse Spatial Intelligence task from the NVIDIA AI City Challenge. For background on the task and the SpatialAgent approach, see:

Hsiang-Wei Huang et al., *Warehouse Spatial Question Answering with LLM Agent: 1st Place Solution of the 9th AI City Challenge Track 3*, ICCV Workshops 2025.

[ICCVW 2025 paper](https://openaccess.thecvf.com/content/ICCV2025W/AICity/papers/Huang_Warehouse_Spatial_Question_Answering_with_LLM_Agent_1st_Place_Solution_ICCVW_2025_paper.pdf)

```bibtex
@InProceedings{Huang_2025_ICCV,
    author    = {Huang, Hsiang-Wei and Kim, Pyongkun and Cheng, Jen-Hao and Chen, Kuang-Ming and Yang, Cheng-Yen and Alattar, Bahaa and Lin, Yi-Ru and Kim, Sangwon and Kim, Kwangju and Huang, Chung-I and Hwang, Jenq-Neng},
    title     = {Warehouse Spatial Question Answering with LLM Agent: 1st Place Solution of the 9th AI City Challenge Track 3},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops},
    month     = {October},
    year      = {2025},
    pages     = {5224--5228}
}
```
