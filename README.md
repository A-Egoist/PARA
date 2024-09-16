# PARA

This is the PyTorch implementation of our paper:

>   Plug-and-play Rating Prediction Adjustment through Trisecting-acting-outcome

![The Workflow of PARA](img/PARA-workflow.svg)

## Requirements

*   numpy=1.21.5
*   pandas=1.3.5
*   python=3.7.16
*   pytorch=1.13.1

## Datasets and Pre-trained Models

| Dataset       | Users   | Items   | Interactions | Link                                                         |
| ------------- | ------- | ------- | ------------ | ------------------------------------------------------------ |
| Amazon-Music  | 478 235 | 266 414 | 836 006      | [URL](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Digital_Music.csv) |
| Ciao          | 17 615  | 16 121  | 72 665       | [URL](https://guoguibing.github.io/librec/datasets.html)     |
| Douban-Book   | 46 548  | 212 995 | 1 908 081    | [URL](https://github.com/DeepGraphLearning/RecommenderSystems/blob/master/socialRec/README.md#douban-data) |
| Douban-Movie  | 94 890  | 81 906  | 11 742 260   | [URL](https://github.com/DeepGraphLearning/RecommenderSystems/blob/master/socialRec/README.md#douban-data) |
| MovieLens-1M  | 6 040   | 3 900   | 1 000 209    | [URL](https://grouplens.org/datasets/movielens/1m/)          |
| MovieLens-10M | 69 878  | 10 677  | 10 000 054   | [URL](https://grouplens.org/datasets/movielens/10m/)         |

https://pan.baidu.com/s/1ZxzcKKBCefl39vp7YKMUng?pwd=8yud

| Method   | MF-based | LightGCN-based | ItemCF-based |
| -------- | -------- | -------------- | ------------ |
| MF       | ✅        | ✅              |              |
| LightGCN |          | ✅              | ✅            |
| ItemCF   |          |                | ✅            |
| IPS      | ✅        |                |              |
| DICE     | ✅        |                |              |
| PDA      | ✅        | ✅              |              |
| TIDE     | ✅        | ✅              |              |
| PARA     | ✅        | ✅              | ✅            |



## Simply Reproduce the Results

Take the Ciao dataset as an example:

1.   Clone source code
     ```bash
     git clone https://github.com/A-Egoist/TWDP.git --depth=1
     ```

2.   Download preprocessed data and pre-trained model at [Baidu Netdisk](https://pan.baidu.com/s/1ZxzcKKBCefl39vp7YKMUng?pwd=8yud)

     -   Each original dataset was re-divided into 5 sets of files according to the requirements of five-fold cross-validation. Each set consists of `*.train`, `*.test`, and `*.extend` files. The `*.train` file is used for training, the `*.test` file for testing, and the `*.extend` file contains data in the format of `(user, positiveItem, negativeItem)`, created by performing negative sampling on the `*.train` file four times. Additionally, the `*.pkl` file stores the processed results of the `*.train` file.
     -   The pre-trained model is named in the format `backbone-method-dataset-fold_index`. For example, if the PARA method uses MF as the backbone model, the models trained on the Ciao dataset would be named `MF-PARA-Ciao-1`, `MF-PARA-Ciao-2`, `MF-PARA-Ciao-3`, `MF-PARA-Ciao-4`, and `MF-PARA-Ciao-5`. The numbering from 1 to 5 indicates the models saved for a five-fold cross-validation experiment.
     -   To reproduce the results of PARA with MF as the backbone model on the Ciao dataset, you need to download the Ciao dataset and ensure it contains the following files: `movie-ratings1.test`, `movie-ratings1.pkl`, `movie-ratings2.test`, `movie-ratings2.pkl`, `movie-ratings3.test`, `movie-ratings3.pkl`, `movie-ratings4.test`, `movie-ratings4.pkl`, and `movie-ratings5.test`, `movie-ratings5.pkl`. Additionally, you need to download the pre-trained models `MF-PARA-ciao-1.pt`, `MF-PARA-ciao-2.pt`, `MF-PARA-ciao-3.pt`, `MF-PARA-ciao-4.pt`, and `MF-PARA-ciao-5.pt`.

3.   Confirm the file tree according to `tree.txt`
     
4.   Inference

     ```bash
     python ./main.py --backbone MF --method PARA --dataset ciao --mode eval
     ```

5.   Convert to Excel
     ```bash
     python ./logs/log_to_excel.py --input ./logs/eval.py --output ./output/eval.xlsx --sl 1 --el 2000
     ```

     

## Start from Scratch

TODO

Take the Ciao dataset as an example:

1.   Clone source code
     ```bash
     ```

2.   Data preprocessing

3.   Training

4.   Inference

5.   Convert to Excel

## Citation

If you use this code, please cite the following paper:

```bibtex
@article{ZhangLong2024PARA,
  title   = {Plug-and-play Rating Prediction Adjustment through Trisecting-acting-outcome},
  author  = {},
  journal = {},
  year    = {},
  volume  = {},
  number  = {},
  pages   = {},
  doi     = {}
}
```

## Acknowledgments

