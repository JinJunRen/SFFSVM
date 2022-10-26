# -*- coding: utf-8 -*-
"""
In this python script we provided an example of how to use our 
implementation of SFFSVM methods to perform classification.

Usage:
```
python demorun.py -data ./dataset/moon_1000_200_2.csv -n 5 
or
python demorun.py -data ./dataset/moon_2000_100_2.csv -n 5
```

run arguments:
    -data : string
    |   Specify a dataset.
    -n : integer
    |   Specify the number of n-fold cross-validation

"""
import sys
sys.path.append("..")
import pandas as pd
import numpy as np
import os
import argparse
import warnings
warnings.filterwarnings("ignore")
from sklearn import svm
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn import preprocessing
import tools.dataprocess as dp
from tools.imbalancedmetrics import ImBinaryMetric
from SlackFactorFSVM import *
RANDOM_STATE = None


def parse():
    '''Parse system arguments.'''
    parse=argparse.ArgumentParser(
        description='General excuting SFFSVM', 
        usage='demorun.py -data <datasetpath> -n <n-fold cross-validation>'
        )
    parse.add_argument("-data",dest="dataset",help="the path of a dataset")
    parse.add_argument("-n",dest="n",type=int,default=5,help="n-fold cross-validation")
    return parse.parse_args()

def metric(y,y_pre):
        return ImBinaryMetric(y,y_pre).AP() 

def validateCSVM(X, y):
        IR=max(np.bincount(y))*1.0/min(np.bincount(y))
        weights = {0:1.0, 1:IR}
        sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2,random_state=RANDOM_STATE)
        C_range = np.logspace(-5, 11, 9,base=2)
        gamma_range = np.logspace(-10, 3, 14,base=2)
        tuned_params = {"gamma":gamma_range,"C" : C_range}
        model = GridSearchCV(svm.SVC(probability=True, class_weight= weights),
                             tuned_params,cv=sss,
                             scoring=make_scorer(metric))#
        model.fit(X,y)
        return model.best_params_

def main():
    para = parse()
    dataset=para.dataset
    scores = []
    X,y=dp.readDateSet(dataset)
    X=preprocessing.scale(X)
    print(f"Dataset:%s,#attribute:%s [neg pos]:%s\n "%(dataset,X.shape[1],str(np.bincount(y))))
    sss = StratifiedShuffleSplit(n_splits=para.n, test_size=0.2,random_state=RANDOM_STATE)
    fcnt=0
    for train_index, test_index in sss.split(X, y):
        fcnt+=1
        print('{} fold'.format(fcnt))
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        #search best parameters of DEC classifier to calculate the slack variables by CV
        print("Searching best parameters (i.e., C and gamma) of  DEC classifier...")
        best_params=validateCSVM(X_train,y_train)
        #search the parameter C of SFSSVM by CV
        print("Searching the parameter (i.e., beta) of SFFSVM...")
        tuned_params={}
        sss2 = StratifiedShuffleSplit(n_splits=5, test_size=0.2)
        tuned_params["beta"]= [(i)*0.1 for i in range(0,11,1)]        
        model = GridSearchCV(SFFSVM(C=best_params['C'], gamma=best_params['gamma']),tuned_params,cv=sss2, n_jobs=-1)
        model.fit(X_train,y_train)
        #predict
        y_pre=model.predict(X_test)
        y_pred = model.predict_proba(X_test)[:, 1]
        metrics=ImBinaryMetric(y_test,y_pre)
        scores.append([metrics.f1(),metrics.MCC(),metrics.aucprc(y_pred)])
        print('F1:{:.3f}\tMMC:{:.3f}\tAUC-PR:{:.3f}'.format(metrics.f1(),metrics.MCC(),metrics.aucprc(y_pred)))
        print('------------------------------')
    # Print results to console
    print('Metrics:')
    df_scores = pd.DataFrame(scores, columns=['F1','MMC','AUC-PR'])
    for metric in df_scores.columns.tolist():
        print ('{}\tmean:{:.3f}  std:{:.3f}'.format(metric, df_scores[metric].mean(), df_scores[metric].std()))
    
if __name__ == '__main__':
    main()
            