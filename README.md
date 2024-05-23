# Joint Moment Retrieval and Highlight Detection Via Natural Language Queries

## Installation

 

- CUDA
- CUDNN
- Python 3.10
- PyTorch 2.0.0
- [NNCore](https://github.com/yeliudev/nncore)
- [ViT-PyTorch](https://github.com/lucidrains/vit-pytorch)


### Install from source

1. Clone the repository from GitHub

```
git clone https://github.com/Skyline-9/Visionary-Vids.git
cd Visionary-Vids
```

2. Install dependencies

Using the shell script (conda required)
```shell
sh environment/init_conda.sh
```

Using conda
```shell
conda env create -f environment/environment.yml
conda activate VisionaryVids
```

Using pip
```shell
pip install -r environment/requirements.txt
```

3. Setup automatic code styling

```shell
pre-commit install
```

## Getting Started

### Download and prepare the datasets

1. Download and extract the datasets.

- [QVHighlights](https://huggingface.co/yeliudev/UMT/resolve/main/datasets/qvhighlights-a8559488.zip)
- [Charades-STA](https://huggingface.co/yeliudev/UMT/resolve/main/datasets/charades-2c9f7bab.zip)
- [YouTube Highlights](https://huggingface.co/yeliudev/UMT/resolve/main/datasets/youtube-8a12ff08.zip)
- [TVSum](https://huggingface.co/yeliudev/UMT/resolve/main/datasets/tvsum-ec05ad4e.zip)

2. Prepare the files in the following structure.

```
Visionary-Vids
├── environment
├── configs
├── datasets
├── models
├── data
│   ├── qvhighlights
│   │   ├── *features
│   │   ├── highlight_{train,val,test}_release.jsonl
│   │   └── subs_train.jsonl
│   ├── charades
│   │   ├── *features
│   │   └── charades_sta_{train,test}.txt
│   ├── youtube
│   │   ├── *features
│   │   └── youtube_anno.json
│   └── tvsum
│       ├── *features
│       └── tvsum_anno.json
├── README.md
├── setup.cfg
├── launch.py
└── ···
```

### Train a model

Run the following command to train a model using a specified config.

```shell
# Single GPU
python launch.py ${path-to-config}

# Multiple GPUs
torchrun --nproc_per_node=${num-gpus} launch.py ${path-to-config}

# Train from checkpoint
python launch.py ${path-to-config} --checkpoint ${path-to-checkpoint}
```

### Test a model and evaluate results

Run the following command to test a model and evaluate results.

```
python launch.py ${path-to-config} --checkpoint ${path-to-checkpoint} --eval
```
