# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 13:27:20 2020

@author: Pedro Oliveira e Heitor Moniz
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

import matplotlib.pyplot as plt



def standertize(Xs):
    Xsmeans = np.mean(Xs, axis = 0)
    Xsstdevs = np.std(Xs, axis = 0)
    res = (Xs-Xsmeans) / Xsstdevs    
    return res


def loadData():
    train = np.loadtxt("TP1_train.tsv", delimiter = '\t')
    test = np.loadtxt("TP1_test.tsv", delimiter = '\t') 
    # SHUFFLE TRAINING SET
    np.random.shuffle(train)  
    # STANDERTIZE TRAINING SET
    x_train = standertize(train[:,:-1])
    y_train = train[:,-1]
    # SHUFFLE TRAINING SET
      
    # STANDERTIZE TEST SET
    x_test = standertize(test[:,:-1]) 
    y_test = test[:,-1]
    np.random.shuffle(test)
    return x_train, y_train, x_test,y_test

def calc_fold(x,y, train,valid,c):
    """return error for train and validation sets"""
    reg = LogisticRegression(C=c, tol=1e-10)
    reg.fit(x[train],y[train])
    prob = reg.predict_proba(x_test)[:,1]
    squares = (prob-y_test)**2
    return np.mean(squares[train]),np.mean(squares[valid]), reg.score(x[valid], y[valid])



x_train, y_train, x_test,y_test = loadData()


folds = 10
kf = StratifiedKFold(n_splits=folds)
e_tr = list ()
e_va = list()
c=10**-2
cx = list()
while(c<10**12):
    cx.append(c)
    for train, valid in kf.split(y_train, y_train):
        tr , va , score = calc_fold(x_train, y_train, train, valid, c)
        print(score)
        
        
        
    e_tr.append(tr)
    e_va.append(va)
    c = 10*c
   

plt.semilogy(e_tr,cx)
plt.semilogy(e_va,cx)
plt.yscale('log')
print(e_tr.index(min(e_tr)))
print(e_va.index(min(e_va)))

   

    

    
    
    
    
