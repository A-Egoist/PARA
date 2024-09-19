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

Dataset information

| Dataset       |   Users |   Items | Interactions |                             Link                             |
| ------------- | ------: | ------: | -----------: | :----------------------------------------------------------: |
| Amazon-Music  | 478 235 | 266 414 |      836 006 | [URL](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Digital_Music.csv) |
| Ciao          |  17 615 |  16 121 |       72 665 |   [URL](https://guoguibing.github.io/librec/datasets.html)   |
| Douban-Book   |  46 548 | 212 995 |    1 908 081 | [URL](https://github.com/DeepGraphLearning/RecommenderSystems/blob/master/socialRec/README.md#douban-data) |
| Douban-Movie  |  94 890 |  81 906 |   11 742 260 | [URL](https://github.com/DeepGraphLearning/RecommenderSystems/blob/master/socialRec/README.md#douban-data) |
| MovieLens-1M  |   6 040 |   3 900 |    1 000 209 |     [URL](https://grouplens.org/datasets/movielens/1m/)      |
| MovieLens-10M |  69 878 |  10 677 |   10 000 054 |     [URL](https://grouplens.org/datasets/movielens/10m/)     |

Pre-trained model information

| Method   | MF-based | LightGCN-based | ItemCF-based |
| -------- | :------: | :------------: | :----------: |
| MF       |    ✅     |                |              |
| LightGCN |          |       ✅        |              |
| ItemCF   |          |                |      ✅       |
| IPS      |    ✅     |                |              |
| DICE     |    ✅     |                |              |
| PDA      |    ✅     |       ✅        |              |
| TIDE     |    ✅     |       ✅        |              |
| PARA     |    ✅     |       ✅        |      ✅       |



## Simply Reproduce the Results

Take the Ciao dataset as an example:

1.   Clone the source code
     ```bash
     git clone https://github.com/A-Egoist/TWDP.git --depth=1
     ```

2.   Download preprocessed data and pre-trained models
     Download the data and models from [Baidu Netdisk](https://pan.baidu.com/s/1ZxzcKKBCefl39vp7YKMUng?pwd=8yud).

     -   Each original dataset is split into 5 sets for five-fold cross-validation, consisting of `*.train`, `*.test`, and `*.extend` files:
         -   `*.train`: used for training.
         -   `*.test`: used for testing.
         -   `*.extend`: includes `(user, positiveItem, negativeItem)` data generate by performing negative sampling on the `*.train` file.
         -   Additionally, the `*.pkl` files store the processed results of the `*.train` files.
         
     -   The pre-trained model files follow the naming convention `backbone-method-dataset-fold_index`. For example, for the PARA method with MF as the backbone model on the Ciao dataset, the models are named as:
         -   `MF-PARA-Ciao-1.pt`
         -   `MF-PARA-Ciao-2.pt`
         -   `MF-PARA-Ciao-3.pt`
         -   `MF-PARA-Ciao-4.pt`
         -   `MF-PARA-Ciao-5.pt`
         
         These files correspond to models trained on different folds for the five-fold cross-validation experiment.
         
     -   To reproduce the results of PARA method with MF as the backbone model on the Ciao dataset, ensure the dataset contains the following files:
         
         -   `movie-ratings1.test`, `movie-ratings1.pkl`
         -   `movie-ratings2.test`, `movie-ratings2.pkl`
         -   `movie-ratings3.test`, `movie-ratings3.pkl`
         -   `movie-ratings4.test`, `movie-ratings4.pkl`
         -   `movie-ratings5.test`, `movie-ratings5.pkl`
         
     -   Additionally, download the corresponding pre-trained models:
         
         -   `MF-PARA-ciao-1.pt`
         -   `MF-PARA-ciao-2.pt`
         -   `MF-PARA-ciao-3.pt`
         -   `MF-PARA-ciao-4.pt`
         -   `MF-PARA-ciao-5.pt`
         
     
3.   Confirm the file structure
     Verify the downloaded files and folder structure according to the provided `tree.txt` file.

4.   Run inference
     To evaluate the model, run the following command:

     ```bash
     # Windows
     python .\main.py --backbone MF --method PARA --dataset ciao --mode eval
     ```
     
     **Avaliable Options:**
     
     *   `--backbone`: The backbone model. Available options: `['MF', 'LightGCN']`.
     *   `--method`: The method to be used. Available options: `['Base', 'IPS', 'DICE', 'PDA', 'TIDE', 'PARA']`.
     *   `--dataset`: The dataset to use. Available options: `['amzoun-music', 'ciao', 'douban-book', 'douban-movie', 'ml-1m', 'ml-10m']`.
     *   `--mode`: The mode to be choosen. Available options: `['train', 'eval', 'both']`.
     
5.   Convert the results to Excel
     After evaluation, convert the log results into an Excel file using the following command:
     
     ```bash
     # Windows
     python .\logs\log_to_excel.py --input .\logs\eval.py --output .\output\eval.xlsx --sl 1 --el 2000
     ```
     
     **Explanation of Parameters:**
     
     *   `--input`: Specifies the log file that contains the evaluation results (e.g., `eval.py`).
     *   `--output`: Specifies the base name of the output Excel file (e.g., `eval.xlsx`). The actual output files will be saved as three separate files:
         *   `eval-mean.xlsx`: Contains the mean of the evaluation metrics.
         *   `eval-std.xlsx`: Contains the standard deviation of the evaluation metrics.
         *   `eval-rank.xlsx`: Contains the ranking of the evaluation metrics.
     *   `--sl`: Specifies the **start line** in the log file from which the results will be converted to Excel.
     *   `--el`: Specifies the **end line** in the log file up to which the results will be converted to Excel.

## Start from Scratch

This section explains how to reproduce the results from scratch, taking the Ciao dataset as an example:

1.   Clone source code
     Clone the repository containing the code:

     ```bash
     git clone https://github.com/A-Egoist/TWDP.git --depth=1
     ```

2.   Data preprocessing

     (a). Split the dataset into 5 sets

     Run the following command to split the Ciao dataset into 5 subsets for five-fold cross-validation:

     ```python
     # Windows
     python .\src\data_processing.py --dataset ciao
     ```

     **Available Option:**

     *   `--dataset`: The dataset to be splited. Available options: `['amzoun-music', 'ciao', 'douban-book', 'douban-movie', 'ml-1m', 'ml-10m']`.

     (b). Compile the negative sampling script

     Use the following command to compile the C++ script for negative sampling:

     ```bash
     # Windows
     g++ .\src\negative_sampling.cpp -o .\src\negative_sampling.exe
     ```

     (c). Perform negative sampling

     Execute the compiled script to perform negative sampling:

     ```bash
     # Windows
     .\src\negative_sampling.exe ciao 1
     ```

     **Explanation of Parameters:**

     *   The **first parameter** specifies the dataset to be processed, with available options: `['amazon-music', 'ciao', 'douban-book', 'douban-movie', 'ml-1m', 'ml-10m']`
     *   The **second parameter** specifies the fold index for the cross-validation dataset, with options: `['1', '2', '3', '4', '5']`

3.   Training
     To start the training process, run the following command:

     ```bash
     # Windows
     python .\main.py --backbone MF --method PARA --dataset ciao --mode train
     ```

     **Available Options:**

     *   `--backbone`: The backbone model. Available options: `['MF', 'LightGCN']`.
     *   `--method`: The method to be used. Available options: `['Base', 'IPS', 'DICE', 'PDA', 'TIDE', 'PARA']`.
     *   `--dataset`: The dataset to use. Available options: `['amzoun-music', 'ciao', 'douban-book', 'douban-movie', 'ml-1m', 'ml-10m']`.
     *   `--mode`: The mode to be chosen. Available options: `['train', 'eval', 'both']`.

4.   Evaluation
     After training, evaluate the model's performance using this command:

     ```bash
     # Windows
     python .\main.py --backbone MF --method PARA --dataset ciao --mode eval
     ```

     Options are the same as step 3.
     
5.   Convert the results to Excel
     Finally, convert the evaluation logs to an Excel file:

     ```bash
     # Windows
     python .\logs\log_to_excel.py --input .\logs\eval.py --output .\output\eval.xlsx --sl 1 --el 2000
     ```

     **Explanation of Parameters:**

     *   `--input`: Specifies the log file that contains the evaluation results (e.g., `eval.py`).
     *   `--output`: Specifies the base name of the output Excel file (e.g., `eval.xlsx`). The actual output files will be saved as three separate files:
         *   `eval-mean.xlsx`: Contains the mean of the evaluation metrics.
         *   `eval-std.xlsx`: Contains the standard deviation of the evaluation metrics.
         *   `eval-rank.xlsx`: Contains the ranking of the evaluation metrics.
     *   `--sl`: Specifies the **start line** in the log file from which the results will be converted to Excel.
     *   `--el`: Specifies the **end line** in the log file up to which the results will be converted to Excel.

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

