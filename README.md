# Source Code and Appendix for "Measuring Item Fairness in Next Basket Recommendation: A Reproducibility Study"
This paper reproduces the fairness metrics implementation and empirical experiments in "Measuring Fairness in Ranked Results: An Analytical and Empirical" https://dl.acm.org/doi/abs/10.1145/3477495.3532018 to investigate whether the lessons about fairness metrics could be generalized to next basket recommendation (NBR).

This repository is built based on the following repositories:

Measuring Fairness in Ranked Results: An Analytical and Empirical. https://github.com/BoiseState/rank-fairness-metrics

A Next Basket Recommendation Reality Check. https://github.com/liming-7/A-Next-Basket-Recommendation-Reality-Check

Based on the above work, we additionally:
* Repreprocess and resplit 3 datasets.
* Tune the hyperparameters of NBR methods and run methods 5 times using 5 random seed.
* Evaluate the performance of NBR methods using 7 fairness metrics and 3 accuracy metrics.
* Investigate the sensitivity of fairness metrics with respective to basket size, position weighting models and user repeat purchase behavior.


## Required packages
To run our data preprocessing and evaluation scripts, Pandas, Numpy and Python >= 3.6 are required.

To run the pubished NBR methods' code, please go to the original repository and check the required packages.

## Contents of this repository
* Source code and datasets.
* Descriptions of different dataset format.
* Pipeline: data preprocessing - run NBR methods - fairness evaluation.
* A PDF file with the additional figures.

## Code structure
* preprocess: contains the script of dataset preprocessing and splitting. 
* csvdata: contains the .csv format dataset after preprocessing.
* jsondata: contains the .json format dataset after preprocessing, history baskets sequence and future basket are stored seperately.
* mergedata: contains the .json format dataset after preprocessing, history baskets sequence and future basket are stored together.
* methods: contains the source code of different NBR methods and the original url repository link of these methods.
* evaluation: scripts for evaluation.
    * fair_metrics: the fairness metrics.
    * metrics.py: the accuracy metrics.
    * model_performance.py: evaluate the fairness and accuracy of recommendation results.
* appendix: contains a PDF file with the additional figures.

## Pipeline
* Step 1. Preprocess and split the datasets. Generate different types of preprocessed datasets for different NBR methods.
* Step 2. Train the model and save the model. (Note that we use the original implementations of the authors, so we provide the original repository links, which contain the instructions of the environment setting, how to run each method, etc. We also provide our additional instructions and hyperparameters in the following section, which can make the running easier.)
* Step 3. Generate the predicted results via the trained model and save the results file.
* Step 4. Use the evaluation scripts to get the performance results.

## Dataset 
### Preprocessing
We provide the scripts of preprocessing, and the preprocessed dataset with different formats (csvdata, jsondata, mergedata), which can be used directly.
If you want to preprocess the dataset yourself, you can download the dataset from the following urls and put them into the "rawdata/{dataset}" folder.
* Instacart: https://www.kaggle.com/c/instacart-market-basket-analysis/data
* Dunnhumby: https://www.dunnhumby.com/source-files/
* Tafeng: https://www.kaggle.com/datasets/chiranjivdas09/ta-feng-grocery-dataset


### Format description of preprocessed dataset
* csvdata: --> G-TopFreq, P-TopFreq, GP-TopFreq, ReCANet
> user_id, order_number, item_id, basket_id

* jsondata: --> TIFUKNN, DNNTSP, DREAM

> history data: {uid1: [[-1], basket, basket, ..., [-1]], uid2:[[-1], basket, basket, ..., [-1]], ... }

> future data: {uid1: [[-1], basket, [-1]], uid2: [[-1], basket, [-1]], ...}

* mergedata: --> UP-CF

> {uid1: [basket, basket, ..., basket], uid2: [basket, basket, ..., basket], ...}

### Format description of predicted results
* Predicted items:

> {uid1: [item, item, ..., item], uid2: [item, item, ..., item], ...}

* Predicted relevance scores (for all the items):

> {uid1: [rel, rel, ..., rel], uid2: [rel, rel, ..., rel], ...}


## Guidelines for each method
Our approach to reproducibility is to rely as much as possible on the artifacts provided by the user themselves, the following repositories have information about how to run each NBR method and the required packages.
* UP-CF@r: https://github.com/MayloIFERR/RACF
* TIFUKNN: https://github.com/HaojiHu/TIFUKNN
* DREAM: https://github.com/yihong-chen/DREAM
* DNNTSP: https://github.com/yule-BUAA/DNNTSP
* ReCANet: https://github.com/mzhariann/recanet

We also provide our additional instructions if the original repository is not clear, as well as the hyperparameters we use.

We set five random seed: 12345, 12321, 54321, 66688, 56789. And the corresponding number of the predicted files are 0, 1, 2, 3, 4.
For G-TopFreq, P-TopFreq, GP-TopFreq, TIFUKNN, the predicted results of each run are same and not influenced by the random seed. Therefore, we only keep one set of predicted files with number 0.

Please create a folder "results" under each method to store the predicted files.

### G-TopFreq, P-TopFreq, GP-TopFreq
Three frequency based methods are under the folder "methods/g-p-gp-topfreq".
* Step 1: Check the file path of the dataset.
* Step 2: Using the following commands to run each method:
```
python g_topfreq.py --dataset instacart 
...
python p_topfreq.py --dataset instacart
...
python gp_topfreq.py --dataset instacart
...
```
Predicted files are stored under folder: "g_top_results", "p_top_results", "gp_top_results".

Predicted file name: {dataset}_pred0.json, {dataset}_rel0.json

### UP-CF@r
UP-CF@r is under the folder "methods/upcf".
* Step 1: Check the dataset path and keyset path.
* Step 2: Predict and save the results using the following commands:
```
python racf.py --dataset instacart --recency 5 --asymmetry 0.25 --locality 5 --seed 12345 --number 0
...
python racf.py --dataset dunnhumby --recency 25 --asymmetry 0.25 --locality 5 --seed 12345 --number 0
...
python racf.py --dataset tafeng --recency 10 --asymmetry 0.25 --locality 1 --seed 12345 --number 0
...
``` 
Predicted file name: {dataset}_pred{number}.json, {dataset}_rel{number}.json

### TIFUKNN
TIFUKNN is under the folder "methods/tifuknn"
* Step 1: Predict and save the results using the following commands:
```
python tifuknn_new.py ../jsondata/instacart_history.json ../jsondata/instacart_future.json ../keyset/instacart_keyset.json 900 0.9 0.6 0.7 3 20 
...
python tifuknn_new.py ../jsondata/dunnhumby_history.json ../jsondata/dunnhumby_future.json ../keyset/dunnhumby_keyset.json 100 0.9 0.9 0.1 7 20 
...
python tifuknn_new.py ../jsondata/tafeng_history.json ../jsondata/tafeng_future.json ../keyset/tafeng_keyset.json 300 0.9 0.9 0.1 11 20 
...
```
Predicted file name: {dataset}_pred0.json, {dataset}_rel0.json

### Dream
Dream is under the folder "methods/dream".
* Step 1: Check the file path of the dataset in the config-param file "{dataset}conf.json"
* Step 2: Train and save the model using the following commands:
```
python trainer.py --dataset instacart --attention 1 --seed 12345 
...
python trainer.py --dataset dunnhumby --attention 1 --seed 12345 
...
python trainer.py --dataset tafeng --attention 1 --seed 12345 
...
```
* Step 3: Predict and save the results using the following commands:
```
python pred_results.py --dataset instacart --attention 1 --seed 12345 --number 0
...
python pred_results.py --dataset dunnhumby --attention 1 --seed 12345 --number 0
...
python pred_results.py --dataset tafeng --attention 1 --seed 12345 --number 0
...
```
Predicted file name: {dataset}_pred{number}.json, {dataset}_rel{number}.json


### DNNTSP
DNNTSP is under the folder "methods/dnntsp".
* Step 1: Confirm the name of config-param file "{dataset}config.json" in ../utils/load_config.py. Check the file path of the dataset in the corresponding file "../utils/{dataset}conf.json". For example:
```
abs_path = os.path.join(os.path.dirname(__file__), "instacartconfig.json")
with open(abs_path) as file:
    config = json.load(file)
```
```
{
    "data": "Instacart",
    "save_model_folder": "DNNTSP",
    "history_path": "../jsondata/instacart_history.json",
    "future_path": "../jsondata/instacart_future.json",
    "keyset_path": "../keyset/instacart_keyset_0.json",
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



### ReCANet
ReCANet is under the folder "methods/recanet"
* Step 1: Predict and save the results using the following commands:
```
python main.py -dataset instacart -user_embed_size 64 -item_embed_size 16 -hidden_size 64 -history_len 35 -number 0 -seed_value 12345
...
python main.py -dataset dunnhumby -user_embed_size 16 -item_embed_size 128 -hidden_size 64 -history_len 35 -number 0 -seed_value 12345 
...
python main.py -dataset tafeng -user_embed_size 64 -item_embed_size 64 -hidden_size 64 -history_len 35 -number 0 -seed_value 12345 
...
```
Predicted file name: {dataset}_pred{number}.json, {dataset}_rel{number}.json


## Evaluation 
Once we got the reommended basket of the model/algorithm on all datasets, you can use our scripts in the evalution folder to evaluate fairness and accuracy of these NBR methods. To apply to NBR scenarios, we made a little modification to the fairness implementation from https://github.com/BoiseState/rank-fairness-metrics

### Performance

* Step 1: Check the dataset, keyset, pred_file path in the code.
* Step 2: Evaluate the performance using the following commands:
```
cd evaluation

python model_performance.py --pred_folder XXX --number_list 01234 --method XXX

```
XXX is the folder where you put the predicted results, number_list corresponds to five numbers of experimental results using five seed.

The fairness and accuracy results will be saved to "results/eval_{method_name}.txt".

