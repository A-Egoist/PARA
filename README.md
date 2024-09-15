# PARA

This is the PyTorch implementation of our paper:

>   Plug-and-play Rating Prediction Adjustment through Trisecting-acting-outcome

![The Workflow of PARA](.\img\PARA-workflow.svg)

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



| Method   | MF-based | LightGCN-based | ItemCF-based |
| -------- | -------- | -------------- | ------------ |
| MF       | URL      |                |              |
| LightGCN |          | URL            |              |
| ItemCF   |          |                | URL          |
| IPS      | URL      |                |              |
| DICE     | URL      |                |              |
| PDA      | URL      | URL            |              |
| TIDE     | URL      | URL            |              |
| PARA     | URL      | URL            | URL          |



## Simply Reproduce the Results

Take the Ciao dataset as an example:

1.   Clone source code
     ```bash
     git clone https://github.com/A-Egoist/TWDP.git --depth=1
     ```

2.   Download preprocessed data

3.   Download pre-trained models

4.   Inference

     ```bash
     ```

5.   Convert to Excel
     ```bash
     ```

     

## Start from Scratch

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

