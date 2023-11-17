# FragSel: Fragmented Selection for Noisy Label Regression

## System Dependencies
- Python >= 3.9
- CUDA >= 9.0 supported GPU

## Installation
Using virtual env is recommended.
```
# create conda env with python=3.9
conda create -n {ENV_NAME} python=3.9

conda activate {ENV_NAME}

# install required version of torch and torchvision
pip install -f https://download.pytorch.org/whl/torch_stable.html torch==1.13.1+cu116 torchvision==0.14.1+cu116

# install other packages
pip install -r requirements.txt
```

## Data and Log directory set-up
Create `checkpoints` and `data` directories.
```
$ mkdir data
$ ln -s [IMDB Data Path] data/imdb_wiki_clean

$ ln -s [log directory path] checkpoints
```

Below, we will guide how to set-up IMDB-clean-bal.
1. Follow the instructions to download https://github.com/yiminglin-ai/imdb-clean, then you will find `[download path]/data/imdb-clean-1024` directory.
2. Convert every images in `data/imdb-clean-1024/` into 128x128 size, and save it to `data/128x128/imdb-clean-1024/` with the same file names.
3. Symlink with main `data` directory as following.
```
$ ln -s [download path]/data data/imdb_wiki_clean
```
4. Copy `imdb_clean_balanced_cnt1000.csv` file into `data/imdb_wiki_clean` directory.

## Run
Specify parameters in `config` yaml, `episodes` yaml files.
```
python main.py --log-dir [log directory path] --config [config file path] --episode [episode file path] --override "|" --random_seed [seed]

```
FragSel IMDB-clean-bal run
```
python main.py --config=configs/imdb_fragment.yaml --episode=episodes/imdb-split4.yaml --log-dir=checkpoints/imdb/dfragment/debug --random_seed= [seed]
```
