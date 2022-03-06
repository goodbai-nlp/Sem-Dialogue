# AMR-Dialogue
An implementation for paper "Semantic Representation for Dialogue Modeling".
You may find our paper [here](https://arxiv.org/pdf/2105.10188).

# Requirements
+ python 3.6
+ pytorch 1.6
+ Tesla V100 (32G)
+ Memory > 150G (for DialogRE)

We recommend to use conda to manage virtual environments:
```
conda create --name <env> --file requirements.txt
```
# Data 
The preprocessed data for DialogRE and DialogRG are avaliable at [here](https://drive.google.com/drive/folders/1UEzGAmV5b7POtlH8sH-rDA0GfXlwteDE?usp=sharing) and [here](https://drive.google.com/file/d/1BciSzIKqezfcxgI3CBWk4Qg_FPCheFpz/view?usp=sharing), respectively.

# Preprocessing
```
bash /path/to/code/preprocess.sh
```

# Training
```
bash /path/to/code/run-dual(hier).sh
```

# Evaluation
```
bash /path/to/code/eval.sh                   # for dialogue relation extraction
bash /path/to/code/decode.sh                 # for dialogue response generation
```

# Pretrained Models

## DialogRE

### Data-v1

|Setting|  dev-F1  | dev-F1c  | test-F1 | test-F1c | checkpoint | 
|  :----:  | :----:  |:---:|  :----:  | :----:  | :----:  | 
| Hier  | 68.3 | 62.9 | 68.4 | 62.3 | [model](https://drive.google.com/file/d/157EpLDMct6HGzWUH36ZCudzjWCNgxh0F/view?usp=sharing) |
| Dual  | 68.6 | 62.6  | 68.0 | 61.5 | [model](https://drive.google.com/file/d/1eNsAEZXMZPGD-WOPyPGEwsysbMLvzL6X/view?usp=sharing) |


### Data-v2

|Setting|  dev-F1  | dev-F1c  | test-F1 | test-F1c | checkpoint |
|  :----:  | :----:  |:---:|  :----:  | :----:  | :----:  |
| Hier  | 68.8 | 62.4  | 66.6 | 61.2 | [model](https://drive.google.com/file/d/14F0YCfBu10S_JV6-vlXHpjGyNJrMZLQC/view?usp=sharing) |
| Dual  | 68.4 | 62.7  | 67.3 | 61.7 | [model](https://drive.google.com/file/d/1oeosWHIva6IWGFvjEYrcPxoGhd1UZ9Kv/view?usp=sharing) |

# Todo
+ clean code


# References
```
@inproceedings{bai-etal-2021-semantic,
    title = "Semantic Representation for Dialogue Modeling",
    author = "Bai, Xuefeng  and
      Chen, Yulong  and
      Song, Linfeng  and
      Zhang, Yue",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.acl-long.342",
    doi = "10.18653/v1/2021.acl-long.342",
    pages = "4430--4445"
}
```
