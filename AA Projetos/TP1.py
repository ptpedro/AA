# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 13:27:20 2020

@author: Pedro Oliveira e Heitor Moniz
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from sklearn.neighbors.kde import KernelDensity



# PRINT MORE VERBOSE
VERBOSE = False
errors = {"training":{"LR": [],"NB":[],"GNB":[]},
        "validation":{"LR": [] , "NB":[],"GNB":[]}}
best_parameters = {"C" : 0, "H" : 0  }
true_error = {"LR" : 0 , "NB" : 0 , "GNB" : 0}
predictions = {"NB" : []}




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
    return x_train, y_train, x_test,y_test, train

folds = 5
x_train, y_train, x_test,y_test, train = loadData()

def calcFoldLogisticRegression(x, y, train_idx, valid_idx, c):
    """return error for train and validation sets"""
    reg = LogisticRegression(C=c)
    reg.fit(x[train_idx],y[train_idx])
    train_error = 1 - reg.score(x[train_idx], y[train_idx])
    validation_error = 1 - reg.score(x[valid_idx], y[valid_idx])
    return train_error, validation_error


def SKFold(folds):
    kf = StratifiedKFold(n_splits=folds)
    return kf


def LogisticRegressionR():
    #criar o objeto de cross
    kf = SKFold(folds)
    cx = list()
    # inicializar a variavel c com o valor mínimo do parametro 'C' da Regressão Logística
    c = 10**-2

    for pow in range  (-2,13):
        cx.append(pow)
        train_error_cumulative=0
        validation_error_cumulative=0
        for train, valid in kf.split(y_train, y_train):
            train_error,validation_error  = calcFoldLogisticRegression(x_train, y_train, train, valid, c)
            train_error_cumulative += train_error
            validation_error_cumulative += validation_error
    
        errors["training"]["LR"].append(train_error_cumulative)
        errors["validation"]["LR"].append(validation_error_cumulative)
        c = 10*c  
    best_parameters["C"] = 10**cx[errors["validation"]["LR"].index(min (errors["validation"]["LR"]))]
    
    
    
def plotLogisticRegression():
    cx =  np.arange(-2, 13, 1)
    plt.figure(figsize=(6, 4))
    plt.plot(cx, errors["training"]["LR"] , '-b', label = 'erro de treino')
    plt.plot(cx, errors["validation"]["LR"] ,'-r', label = 'validação')
    plt.xlabel('log base 10')
    plt.ylabel('erro de treino')
    plt.legend()
    plt.savefig("LR.png")
    plt.show()


        








#First we have to startify: kf = StratifiedKFold(n_splits=folds) where folds = 5. 
#Then we need a loop for the iteration of the H parameter, which goes from 0.02 to 0.6 with step 0.02 (according to the assignment). 
#Then, for each iteration of H, we make kf.split to obtain a validation and a training 
#set from the initial X_r data (not using the test file data, of course). 
#And then obtain the validation and the training error for the current H value (we use to call this the calc_fold). 
#In the calc_fold we have to separate the training data in subsets, each one belonging to each class. 
#Then for each class and each attribute we have to call the kde.fit for obtaining each attribute density
#for that class. 
#Then using the kde.score_samples we obtain the logarithm of p(xi|class).
#This logarithm values must be summed  for every attribute for the same class. 
#Then add the logarithm of the apriori probability of that class.#
#The class having the highest logarithm of the probability is the class that each object belongs

def calcFoldNB(x_train,y_train,train,valid,h):
    training = [x_train[y_train == 0], x_train[y_train == 1]]
    trainingfit = [x_train[train][y_train[train] == 0],x_train[train][y_train[train] == 1]  ]
    classifiers0 = list()
    classifiers1 = list()
    logaAPRIO = [np.log(training[0].shape[0] / y_train.shape[0]),
                np.log(training[1].shape[0] / y_train.shape[0])]
    for atr in range(x_train.shape[1]):
        classifiers0.append(KernelDensity(bandwidth=h).fit(trainingfit[0][:,[atr]]))
        classifiers1.append(KernelDensity(bandwidth=h).fit(trainingfit[1][:,[atr]]))                     
    preds = np.zeros((x_train.shape[0],2))
    for i in range(len(classifiers0)):
        preds[:,1] += classifiers0[i].score_samples(x_train[:,[i]])
    preds[:,1] += logaAPRIO[0]
    for b in range(len(classifiers1)):  
        preds[:,1] -= classifiers1[b].score_samples(x_train[:,[b]])
    preds[:,1] -= logaAPRIO[1]
    preds[preds[:,1]<0,0]=1
    return np.sum( preds[train,0] != y_train[train] ) / y_train[train].shape[0] , np.sum(preds[valid,0] != y_train[valid] ) / y_train[valid].shape[0]

def calcPreds(xs,ys,x_test,y_test,h):   
    training = [xs[ys == 0], xs[ys == 1]]
    classifiers0 = list()
    classifiers1 = list()
    logaAPRIO = [np.log(training[0].shape[0] / ys.shape[0]),
                np.log(training[1].shape[0] / ys.shape[0])]
    for atr in range(xs.shape[1]):
        classifiers0.append(KernelDensity(bandwidth=h).fit(training[0][:,[atr]]))
        classifiers1.append(KernelDensity(bandwidth=h).fit(training[1][:,[atr]]))                     
    preds = np.zeros((x_test.shape[0],2))
    for i in range(len(classifiers0)):
        preds[:,1] += classifiers0[i].score_samples(x_test[:,[i]])
    preds[:,1] += logaAPRIO[0]
    for b in range(len(classifiers1)):  
        preds[:,1] -= classifiers1[b].score_samples(x_test[:,[b]])
    preds[:,1] -= logaAPRIO[1]
    preds[preds[:,1]<0,0]=1
    predictions["NB"] = preds[:,0]
    return np.sum( preds[:,0] != y_test ) / y_test.shape[0] , preds
    
    

def NB():
    kf = StratifiedKFold(n_splits=folds)
    for h in np.arange(0.02, 0.62, 0.02):
        train_error_cumulative=0
        validation_error_cumulative=0
        for train, valid in kf.split(y_train,y_train):
            train_error, validation_error = calcFoldNB(x_train, y_train,train,valid,h)
           
          
        train_error_cumulative += train_error/folds
        validation_error_cumulative += validation_error/folds
        errors["training"]["NB"].append(train_error_cumulative)
        errors["validation"]["NB"].append(validation_error_cumulative)
        best_parameters["H"] =  0.02*errors["validation"]["NB"].index(min( errors["validation"]["NB"]))
   
       


def plotNB():
    plt.figure(figsize=(6, 4))
    plt.plot(np.arange(0.02,0.62,0.02), errors["training"]["NB"], '-b', label = 'NB training error')
    plt.plot(np.arange(0.02,0.62,0.02), errors["validation"]["NB"], '-r', label = 'validation')
    plt.xlabel('bandwidth')
    plt.ylabel('error')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    plt.legend()
    plt.savefig("NB.png")
    plt.show()
   



def GaussianNBSKLearn(x_train,y_train):
    GNBClf = GaussianNB()
    return GNBClf.fit(x_train, y_train)  

def calculateTrueErrorClassifiers():
    lr = LogisticRegression(C = best_parameters["C"]).fit(x_test,y_test)
    true_error["LR"] = 1- lr.score(x_test,y_test)
    true_error["NB"]  = calcPreds(x_train,y_train,x_test,y_test,best_parameters["H"])[0]
    gnb = GaussianNBSKLearn(x_train,y_train)
    true_error["GNB"] = 1-gnb.score(x_test,y_test)
    

def normalTest():
    n = x_test.shape[0]
    x_lr = true_error["LR"] * n
    P0_lr = x_lr / n
    sig_lr = np.sqrt(n * P0_lr * (1-P0_lr))
    print("Logistic Regression approximate normal test")
    print("X +- 1,96sig = ", x_lr , "+-", 1.96 * sig_lr, "\n")
    x_nb =  true_error["NB"] * n
    P0_nb = true_error["NB"]
    sig_nb = np.sqrt(n * P0_nb * (1 - P0_nb))
    print("Own Naive Bayes approximate normal test")
    print("X +- 1,96sig = ", x_nb , "+-", 1.96 * sig_nb, "\n")
    x_gnb = true_error["GNB"] * n
    P0_gnb = true_error["GNB"]
    sig_gnb = np.sqrt(n * P0_gnb * (1 - P0_gnb))
    print("Scikit learn Naive Bayes approximate normal test")
    print("X +- 1,96sig = ", x_gnb , "+-", 1.96 * sig_gnb, "\n")




    
        
LogisticRegressionR()
plotLogisticRegression()
NB()
plotNB()
calculateTrueErrorClassifiers()
normalTest()



#McNemar 

#LR vs NB
lr = LogisticRegression()
lr_fit = lr.fit(x_train,y_test)
lr_predictions = lr_fit.predict(x_test)
miss_classified_LR = np.where(lr_predictions - y_test != 0)[0]
good_classified_LR = np.where(lr_predictions - y_test == 0)[0]
miss_classified_NB = np.where( predictions["NB"] - y_test != 0)[0] 
good_classified_NB = np.where(predictions["NB"] - y_test == 0)[0]
e01 = len(np.intersect1d(miss_classified_NB, good_classified_LR)) 
e10 = len(np.intersect1d(good_classified_NB, miss_classified_LR)) 
print("McNemar’s test SVM Vs Own Naive Bayes")
print("LR VS own Naive Bayes = ",((np.abs(e01-e10) - 1)**2) / (e01+e10),"\n")

#Own NB vs GNB
gnb_fit = GaussianNBSKLearn(x_train,y_train)
gnb_predictions = gnb_fit.predict(x_test)
miss_classified_GNB = np.where( gnb_predictions - y_test != 0)[0] 
good_classified_GNB = np.where( gnb_predictions - y_test == 0)[0]
e01 = len(np.intersect1d(miss_classified_NB, good_classified_GNB)) 
e10 = len(np.intersect1d(good_classified_NB, miss_classified_GNB)) 
print("McNemar’s test GNB Vs Own Naive Bayes")
print("GNB VS own NB = ",((np.abs(e01-e10) - 1)**2) / (e01+e10),"\n")

#GNB vs LR
e01 = len(np.intersect1d(miss_classified_GNB, good_classified_LR)) 
e10 = len(np.intersect1d(good_classified_GNB, miss_classified_LR)) 
print("McNemar’s test GNB Vs Own Naive Bayes")
print("GNB VS own LR = ",((np.abs(e01-e10) - 1)**2) / (e01+e10),"\n")

    
    
    
    


