# Source Code and Appendix for "Measuring Item Fairness in Next Basket Recommendation: A Reproducibility Study"


## Required packages
To run our data preprocessing, evaluation scripts, Pandas, Numpy and Python >= 3.6 are required.

To run the pubished methods' code, you can go to the original repository and check the required packages.
## Contents of this repository
* Source code and datasets.
* Descriptions of different dataset format.
* Pipelines about how to run and get results.
* A PDF file with the additional plots.

## Structure
* preprocess: contains the script of dataset preprocessing. 
* csvdata: contains the .csv format dataset after preprocessing.
* jsondata: contains the .json format dataset after preprocessing, history baskets sequence and future basket are stored seperately.
* mergedata: contains the .json format dataset after preprocessing, history baskets sequence and future basket are stored together.
* methods: contains the source code of different NBR methods and the original url repository link of these methods.
* evaluation: scripts for evaluation.
    * fair_metrics: the fair ranking metrics.
    * metrics.py: the accuracy metrics.
    * model_performance.py: evaluate the fairness and accuracy of recommendation results.
* appendix: contains a PDF file with the additional plots.

## Pipeline
* Step 1. Select the different types of preprossed datasets according to different methods. (Edit the entry or put datasets at the corresponding folder.)
* Step 2. Train the model and save the model. (Note that we use the original implementations of the authors, so we provide the original repository links, which contain the instructions of the environment setting, how to run each method, etc. We also provide our additional instructions in the following section, which can make the running easier.)
* Step 3. Generate the predicted results via the trained model and save the results file.
* Step 4. Use the evaluation scripts to get the performance results.

## Dataset 
### Preprocessing
We provide the scripts of preprocessing, and the preprocessed dataset with different formats, which can be used directly.
If you want to preprocess the dataset yourself, you can download the dataset from the following urls:
* Instacart: https://www.kaggle.com/c/instacart-market-basket-analysis/data
* Dunnhumby: https://www.dunnhumby.com/source-files/
* Tafeng: https://www.kaggle.com/datasets/chiranjivdas09/ta-feng-grocery-dataset


### Format description of preprocessed dataset
* dataset: --> G-TopFreq, P-TopFreq, GP-TopFreq, ReCANet
> csv format
* jsondata: --> TIFUKNN, DNNTSP, DREAM

> history data: {uid1: [[-1], basket, basket, ..., [-1]], uid2:[[-1], basket, basket, ..., [-1]], ... }

> future data: {uid1: [[-1], basket, [-1]], uid2: [[-1], basket, [-1]], ...}

* mergedata: --> UP-CF

> {uid1: [basket, basket, ..., basket], uid2: [basket, basket, ..., basket], ...}

* Predicted results:

> {uid1: [item, item, ..., item], uid2: [item, item, ..., item], ...}

* Predicted relevance scores (for all the items):

> {uid1: [rel, rel, ..., rel], uid2: [rel, rel, ..., rel], ...}


## Guidelines for each method
Our approach to reproducibility is to rely as much as possible on the artifacts provided by the user themselves, the following repositories have information about how to run each NBR method and the required packages.
* DREAM: https://github.com/yihong-chen/DREAM
* DNNTSP: https://github.com/yule-BUAA/DNNTSP
* TIFUKNN: https://github.com/HaojiHu/TIFUKNN
* UP-CF@r: https://github.com/MayloIFERR/RACF
* ReCANet: https://github.com/mzhariann/recanet

We also provide our additional instructions if the original repository is not clear, as well as the parameters we use.


### G-TopFreq, P-TopFreq, GP-TopFreq
Three frequency based methods are under the folder "methods/g-p-gp-topfreq".
* Step 1: Check the file path of the dataset, or put the dataset into corresponding folder.
* Step 2: Using the following commands to run each method:
```
python g_topfreq.py --dataset dunnhumby 
...
python p_topfreq.py --dataset dunnhumby 
...
python gp_topfreq.py --dataset dunnhumby
...
```
Predicted files are stored under folder: "g_top_results", "p_top_results", "gp_top_results".

Predicted file name: {dataset}_pred.json, {dataset}_rel.json

### Dream
Dream is under the folder "methods/dream".
* Step 1: Check the file path of the dataset in the config-param file "{dataset}conf.json"
* Step 2: Train and save the model using the following commands:
```
python trainer.py --dataset dunnhumby --attention 1
...
python trainer.py --dataset tafeng --attention 1
...
python trainer.py --dataset instacart --attention 1
...
```
* Step 3: Predict and save the results using the following commands:
```
python pred_results.py --dataset dunnhumby --attention 1 --seed 12345 --number 0
...
python pred_results.py --dataset tafeng --attention 1 --seed 12345 --number 0
...
python pred_results.py --dataset instacart --attention 1 --seed 12345 --number 0
...
```
Predicted file name: {dataset}_pred{number}.json, {dataset}_rel{number}.json


### DNNTSP
DNNTSP is under the folder "methods/dnntsp".
* Step 1: Go to config/parameter file, edit the following values: data, history_path, future_path, keyset_ path, item_embed_dim, items_total ... an example:
```
{
    "data": "Instacart",
    "save_model_folder": "DNNTSP",
    "history_path": "../jsondata/instacart_history.json",
    "future_path": "../jsondata/instacart_future.json",
    "keyset_path": "../keyset/instacart_keyset.json",
    "items_total": 29399,
    "item_embed_dim": 16,
    "cuda": 0,
    "loss_function": "multi_label_soft_loss",
    "epochs": 40,
    "batch_size": 64,
    "learning_rate": 0.001,
    "optim": "Adam",
    "weight_decay": 0
}
```
* Step 2: Train and save the models using the following command:
```
python train_main.py --seed 12345
```
* Step 3: Predict and save results using the following commands:
```
python pred_results.py --dataset instacart --number 0 --best_model_path XXX
```
Note, DNNTSP will save several models during the training, an epoch model will be saved if it has higher performance than previous epoch, so XXX is the path of the last model saved during the training.

Predicted file name: {dataset}_pred{number}.json, {dataset}_rel{number}.json
### UP-CF
UP-CF is under the folder "methods/upcf".
* Step 1: Copy the dataset to its folder, and check the dataset path and keyset path.
* Step 2: Predict and save the results using the following commands:
```
python racf.py --dataset dunnhumby --foldk 0 --recency 25 --asymmetry 0.25 --locality 5
...
python racf.py --dataset tafeng --foldk 0 --recency 10 --asymmetry 0.25 --locality 1
...
python racf.py --dataset instacart --foldk 0 --recency 5 --asymmetry 0.25 --locality 5
...
``` 
Predicted file name: {dataset}_pred{number}.json, {dataset}_rel{number}.json

### TIFUKNN
TIFUKNN is under the folder "methods/tifuknn"
* Step 1: Predict and save the results using the following commands:
```
cd tifuknn
python tifuknn_new.py ../../jsondata/dunnhumby_history.json ../../jsondata/dunnhumby_future.json ../../keyset/dunnhumby_keyset_0.json 900 0.9 0.6 0.2 3 20
...
python tifuknn_new.py ../../jsondata/tafeng_history.json ../../jsondata/tafeng_future.json ../../keyset/tafeng_keyset_0.json 300 0.9 0.7 0.7 7 20
...
python tifuknn_new.py ../../jsondata/instacart_history.json ../../jsondata/instacart_future.json ../../keyset/instacart_keyset_0.json 900 0.9 0.7 0.9 3 20
```
Predicted file name: {dataset}_pred{foldk}.json

## Evaluation 
Once we got the reommended basket of the model/algorithm on all datasets, you can use our scripts in the evalution folder to evaluate performance w.r.t. repetition and exploration.

Note that, each method will save their results to their own pred folder. 

### Performance

* Step 1: Check the dataset, keyset, pred_file path in the code.
* Step 2: Evaluate the performance using the following commands:
```
cd evaluation
python model_performance.py --pred_folder XXX --fold_list [0, 1, 2, ...]
```
XXX is the folder where you put the predicted baskets, fold_list requires a list of all the keyset files you use in the experiments.

The results will be printed out in the terminal and saved to "eval_results.txt".

### Performance gain
* Step 1: Check the dataset, keyset, pred_file path in the code.
* Step 2: Evaluate the performance using the following commands:
 ```
cd evaluation
python performance_gain.py --pred_folder XXX --fold_list [0, 1, 2, ...]
```
XXX is the folder where you put the predicted baskets, fold_list requires a list of all the keyset files you use in the experiments.

The results will be printed out in the terminal.
