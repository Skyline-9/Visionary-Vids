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

- [QVHighlights](https://connectpolyu-my.sharepoint.com/:u:/g/personal/21039533r_connect_polyu_hk/EVyfPQmNEfxCpvWO3Lp-6NkBld4GHGH8sPj1ZVkx4ScKNg?e=LRS0gQ)
- [Charades-STA](https://connectpolyu-my.sharepoint.com/:u:/g/personal/21039533r_connect_polyu_hk/EXq0dTx1exhBimH1S4JDqtoBt2hj2gC3tazWHMMaBDNK8Q?e=9pIeav)
- [YouTube Highlights](https://connectpolyu-my.sharepoint.com/:u:/g/personal/21039533r_connect_polyu_hk/EWv-_88eTGZJr0VwUp51NbABbcQe8BBM4VWOipghje79aQ?e=MbJpgn)
- [TVSum](https://connectpolyu-my.sharepoint.com/:u:/g/personal/21039533r_connect_polyu_hk/ESH3Wx6l-kBGmRvf2dfplesBaq4SJp9SxKyYypEO7UKVOA?e=1Naroo)

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
torchrun --nproc_per_node=${num-gpus} tools/launch.py ${path-to-config}
```

### Test a model and evaluate results

Run the following command to test a model and evaluate results.

```
python launch.py ${path-to-config} --checkpoint ${path-to-checkpoint} --eval
```