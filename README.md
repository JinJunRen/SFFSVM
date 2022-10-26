# Slack-Factor-based Fuzzy Support Vector Machine(briefly SFFSVM)

We design a new fuzzy membership function and combine it with cost-sensitive learning to deal with the class imbalance problem with noise samples, named SFFSVM. In SFFSVM, the relative distances between samples and an estimated hyperplane, called slack factors, are used to define the fuzzy membership function. To eliminate the impact of class imbalance on the function and gain more accurate samples' importance, we rectify the importance according to both the positional relationship between the estimated hyperplane and the optimal hyperplane of the problem and the slack factors of samples. Comprehensive experiments on artificial and real-world datasets demonstrate that SFFSVM outperforms other comparative methods on F1, MCC, and AUC-PR metrics. 

# Cite us

If you find this repository helpful in your work or research, we would greatly appreciate citations to the following paper:
```

```

# Install

Our SFFSVM implementation requires following dependencies:
- [python](https://www.python.org/) (>=3.7)
- [numpy](https://numpy.org/) (>=1.11)
- [scipy](https://www.scipy.org/) (>=0.17)
- [scikit-learn](https://scikit-learn.org/stable/) (>=0.21)


```
git clone https://github.com/JinJunRen/SFFSVM
```

# Usage

## Documentation
**SlackFactorFSVM.py**

| Parameters    | Description   |
| ------------- | ------------- |
| `C`    | *float, optional (default=100)* <br>  Regularization parameter. The strength of the regularization is inversely proportional to C. Must be strictly positive. <br>  The penalty is a squared l2 penalty. |
| `gammar`    | *float, optional (default=auto)* <br>  The  kernel width of rbf. |
| `beta`    | *float, optional (default=0)* <br>  Smoothing parameter. Its interval is [0,1] |
----------------

| Methods    | Description   |
| ---------- | ------------- |
| `fit(self, X, y)` | Build a SFFSVM classifier on the training set (X, y).|
| `predict(self, X)` | Predict class for X. |
| `predict_proba(self, X)` | Predict class probabilities for X. |
| `score(self, X, y)` | Return the average precision score on the given test data and labels. |
| `calcKxi(self,X,y)` | Calcuate the slack variables of samples X. |
| `calc_dec_fun(self, clf,X)` | calcuate the value of decision function of samples X. |


----------------

**demorun.py**

In this python script we provided an example of how to use our implementation of SFFSVM methods to perform classification.

| Parameters    | Description   |
| ------------- | ------------- |
| `data` | *String*<br> Specify a dataset. |
| `n`  | *Integer,(default=`5`)*<br> Specify the number of n-fold cross-validation. |

----------------

## Examples

```python
python demorun.py -data ./dataset/moon_1000_200_2.csv -n 5 
or
python demorun.py -data ./dataset/moon_2000_100_2.csv -n 5
```

##Dataset links:
[Knowledge Extraction Based on Evolutionary Learning (KEEL)](https://sci2s.ugr.es/keel/studies.php?cat=imb).
