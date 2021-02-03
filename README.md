# AutoEncoder
3-layer simple Auto Encoder without specific ML library written in Python3


---

## Structure

| Name              | Number                    | Recommended   |
| --                | --                        | --            |
| hidden layer      | 1                         | 1             |
| total layer       | 3                         | 3             |
| input layer node  | arbitrary                 | 10 ~ 50       |
| hidden layer node | arbitrary                 | 5 ~ 25        |
| output layer node | same to input layer node  | 10 ~ 50       |


## Preliminary 

This program requires modules below:

- numpy
- pandas
- matplotlib
- joblib (if you want to parallel ***trials***)


> TASK: **to optimize weight vector to make sure that the input and output of the auto encoder are equal. The input is 100 binary lists with the length of ***the number of input layer nodes*** for both training and predicting(test).**


## Python files

to run NN

| File name             | Class         | Explanation                           |
| --                    | --            | --                                    |
| `main.py`             | (None)        | to run Autoencoder                    |
| `auto_encoder.py`     | AE            | program body                          |
| `configuration.py`    | Configuration | to set hyperparameters and dataset    |
| `logger.py`           | Logger        | to export log                         |


to summarize and compare results collected while changing hyperparameters(e.g., the number of input/hidden layer)

| File name             | Class         | Explanation                           |
| --                    | --            | --                                    |
| `summarizer.py`       | Summary       | to summarize and compare              |
| `config_sum.py`       | Configuration | to select data and color to use       | 



## Output and folders and files

exported by `logger.py`

| Name                          | Folder/File   | Explanation                                                                   |
| --                            | --            | --                                                                            |
| `trial`                       | Folder        | where result of each trial stored (values of nodes, prediction probability)   |
| `all_trials.csv`              | File          | error of classification of each epochs in each trial                          |
| `experimental_setting.txt`    | File          | log of configuration                                                          |
| `statistics.csv`              | File          | *min, q25, med, q75, max, ave, std of accuracy of whole trials                |

exported by `summarizer.py`

| Name                          | Explanation                           |
| --                            | --                                    |
| `compare_AE_*.png`            | figure to compare convergence speed   |

---