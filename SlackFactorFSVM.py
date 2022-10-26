# -*- coding: utf-8 -*-
"""
Created on  Sep  7 14:44:15 2022

@author: Jinunren
mailto: jinjunren@lzufe.edu.cn
"""

import numpy as np
from sklearn import svm
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, ClassifierMixin
from tools.imbalancedmetrics import ImBinaryMetric
from sklearn.metrics import make_scorer


class BaseSFFSVM(BaseEstimator, ClassifierMixin):
    def __init__(self,gamma='auto'):
        self.gamma=gamma

    def replaceLable(self, y, neg_label=0, pos_label=1):
        '''
		if y_i is the positive class then we assigin to +1 to it, and
		if y_i is the negative class then we assigin to +1 to it.
        '''
        newy=np.copy(y)
        newy[newy==neg_label]=-1.0 
        newy[newy==pos_label]=1.0    
        return newy
    
    def calc_dec_fun(self, clf,X):
        '''
		calcuate the value of decision function of samples X 
        '''
        return np.abs(clf.decision_function(X))
    
    def calcKxi(self, clf,X,y):
        '''
		calcuate the slack variables of samples X 
        '''
        y_new=self.replaceLable(y)
        dec_fun=clf.decision_function(X)
        Kxi= 1- y_new * dec_fun.reshape(-1)
        Kxi[Kxi<0] = 0
        return Kxi

    @classmethod
    def metric(self,y,y_pre):
        return ImBinaryMetric(y,y_pre).AP()
     

class SFFSVM(BaseSFFSVM):
    def __init__(self, C=100, gamma='auto',beta=0):
        super(SFFSVM,self).__init__(gamma)
        self.C=C
        self.beta=beta
        self.gamma = gamma

    def fit(self, X, y):
        IR=max(np.bincount(y))*1.0/min(np.bincount(y))
        weights = {0:1.0, 1:IR}
        model=svm.SVC(C=self.C, gamma=self.gamma, probability=True, class_weight= weights)#
        model.fit(X,y)
        #calculate the slack Factors of the samples
        kxi = self.calcKxi(model, X , y)
        pos_M_ind=np.intersect1d(np.where((kxi>0)&(kxi<=1))[0],np.where(y==1)[0])
        pos_E_ind=np.intersect1d(np.where(kxi>1)[0],np.where(y==1)[0])
        neg_E_ind=np.intersect1d(np.where(kxi>2)[0],np.where(y==0)[0])
        del_ind=list(pos_E_ind)
        samples_weight=np.ones(len(kxi))
        if len(pos_M_ind)>0:
            samples_weight[pos_M_ind] = (2.0/ (np.exp(kxi[pos_M_ind] * self.beta)+1))
        if len(neg_E_ind)>0:
            samples_weight[neg_E_ind] = np.exp(-1* kxi[neg_E_ind] * self.beta)
        samples_weight[del_ind] = 1e-5    
        model.fit(X, y, samples_weight)
        self.clf = model
        return self
    
    
    def predict_proba(self, X , y= None):
        return self.clf.predict_proba(X)
    
    def predict(self, X , y= None):
        return self.clf.predict(X)
    
    def decision_function(self, X):
        return self.clf.decision_function(X)
    
    def score(self, X, y):
        y_pre=self.predict(X)
        return self.metric(y,y_pre)
    
