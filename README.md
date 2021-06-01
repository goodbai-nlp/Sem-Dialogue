# AMR-Backparsing
An implementation for paper "Semantic Representation for Dialogue Modeling" (to appear in ACL 2021)

# Requirements
+ python 3.6
+ pytorch 1.6

We recommend to use conda to manage enviroments:
```
conda create --name <env> --file requirements.txt
```
# Data 
We provide preprocessed data for two tasks here [tobeadded].

# Preprocessing
```
bash /path/to/code/preprocess.sh
```

# Training
```
bash /path/to/code/run-dual/hier.sh
```

# Evaluation
```
bash /path/to/code/eval.sh                   # for dialogue relation extraction
bash /path/to/code/decode.sh                 # for dialogue response generation
```

# Todo
+ upload preprocessed data
+ clean code

# References
```
@inproceedings{bai-etal-2020-online,
    title = "Semantic Representation for Dialogue Modeling",
    author = "Bai, Xuefeng  and 
      Chen, Yulong and
      Song, Linfeng  and
      Zhang, Yue",
    booktitle = "Proceedings of the Joint Conference of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (ACL-IJCNLP 2021)
",
    month = August,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "todo",
    doi = "todo",
    pages = "todo",
}
```
