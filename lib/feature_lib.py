import numpy as np
import pandas as pd
import random

from sklearn import preprocessing
from sklearn.model_selection import train_test_split, KFold,cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import f1_score,confusion_matrix
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier


def init_model_list():
    '''
    No Input
    
    Output : model list
    '''
    models = []
    
    models.append(('LR', LogisticRegression(solver='lbfgs')))
    models.append(('RF', RandomForestClassifier()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('SVM', SVC(gamma='auto')))
    models.append(('LSVM', LinearSVC()))
    models.append(('GNB', GaussianNB()))
    models.append(('DTC', DecisionTreeClassifier()))
    models.append(('GBC', GradientBoostingClassifier()))
    models.append(('XG', XGBClassifier()))
    
    acc_lists = [[] for i in range(len(models))]
    
    return models,acc_lists


def random_feature_selection(feature, num_feature):
    '''
    Params
        feature
            feature list
            dtype : pandas object
            
        num_feature
            determine the number of feature that randomly selected
            dtype : int
    
    Output
        random_features
            randomly selected features
    '''
    data_feature = list(feature)
    random_features = random.sample(data_feature, num_feature)
    return random_features


def feature_random_test(data, label,num_iter = 100):
    '''
    
    
    '''
    model_list,acc_list = init_model_list()
    
    iter_idx = 0
    iter_acc = np.zeros((len(model_list), num_iter))
    
    data_feature = list(data.columns)
    acc_list_mean = np.zeros((len(data_feature),len(model_list)))
    acc_list_std = np.zeros((len(data_feature),len(model_list)))
    
    feature_len = len(data_feature)
    for num_feature in range(feature_len):
        print('[Number of Feature] : ',num_feature)
        iter_idx = 0
        while iter_idx != num_iter:
            random_feature = random_feature_selection(data_feature,num_feature+1)
            #print(random_feature)
            x_selected = data[random_feature]
            x_train, x_test, y_train, y_test = train_test_split(x_selected,label, test_size=0.2, random_state=42)
            
            idx = 0
            for name, model in model_list:
                model.fit(x_train, y_train)
                model_acc = accuracy_score(y_test, model.predict(x_test))
                
                iter_acc[idx][iter_idx] = model_acc

                idx = idx+1

            iter_idx = iter_idx +1
        
        for i in range(len(model_list)):
            model_mean = np.mean(iter_acc[i])
            model_std = np.std(iter_acc[i])
            acc_list_mean[num_feature][i] = model_mean
            acc_list_std[num_feature][i] = model_std
            
    return model_list,acc_list_mean, acc_list_std



def minmax_selector(data, n, initial):
    '''
    Input
        data
        n
        initial
    
    '''
    #data = (data-data.mean())/data.std()
    data = data
    dissmat = pd.DataFrame(np.arccos(np.corrcoef(data.values.T)),index=data.columns,columns=data.columns)
    result = [initial]
    n-=1
    while(n!=0):
        n -= 1
        result += [dissmat[result].min(axis=1).idxmax()]
    return result


def feature_minmax_test(data, label,dominant_features):
    '''
    Find optimized featrue
    Input
        data 
        label
    Output
        1. optimized feature number
        2. Optimized feature plot
    
    '''
    model_list,acc_list = init_model_list()
 

    for i in range(len(data.columns)):
        select_feature_list = minmax_selector(data,i+1,dominant_features)
        x_selected = data[select_feature_list]
        x_train, x_test, y_train, y_test = train_test_split(x_selected,label, test_size=0.2, random_state=42)
        
        idx = 0
        for name, model in model_list:
            model.fit(x_train, y_train)
            #kfold = KFold(n_splits=5, random_state=7)
            #accuracy_results =cross_val_score(model, x_train,y_train, cv=kfold, scoring='accuracy')
            #acc_list[idx].append(accuracy_results)
            model_acc = accuracy_score(y_test, model.predict(x_test))
            acc_list[idx].append(model_acc)
            
            #print(accuracyMessage) 
            idx = idx+1
            
    return model_list,acc_list
    
'''
Cluster selection

'''

def feature_cluster_test(data, label):
    '''
    Find optimized featrue
    Input
        data 
        label
    Output
        1. optimized feature number
        2. Optimized feature plot
    
    '''
    model_list,acc_list = init_model_list()
    acc_list = np.zeros(len(model_list))

    x_selected = data
    x_train, x_test, y_train, y_test = train_test_split(x_selected,label, test_size=0.2, random_state=42)

    idx = 0
    for name, model in model_list:
        model.fit(x_train, y_train)
        #kfold = KFold(n_splits=5, random_state=7)
        #accuracy_results =cross_val_score(model, x_train,y_train, cv=kfold, scoring='accuracy')
        #acc_list[idx].append(accuracy_results)
        model_acc = accuracy_score(y_test, model.predict(x_test))
        #acc_list[idx].append(model_acc)
        acc_list[idx] = model_acc
        #print(accuracyMessage) 
        idx = idx+1
            
    return model_list,acc_list


def myslink(dist):
    N = len(dist)
    p = np.zeros(N, dtype = int)
    l = np.zeros(N, dtype = float)
    m = np.zeros(N, dtype = float)
    
    for n in np.arange(N):
        # S1
        p[n] = n
        l[n] = np.inf
        
        # S2
        m[:n] = dist[:n,n]
        
        # S3
        for i in np.arange(n):
            if l[i] >= m[i]:
                m[p[i]] = np.min((m[p[i]],l[i]))
                l[i] = m[i]
                p[i] = n
            else:
                m[p[i]] = np.min((m[p[i]],m[i]))
                
        # S4
        for i in np.arange(n):
            if l[i] >= l[p[i]]:
                p[i] = n
                
    return p,l


def myclustering(points, epsilon,metrictype='arcmetric'):
    n = 30
    if metrictype == 'arcmetric':
        dist = np.arccos(np.corrcoef(points.values.T))
    elif metrictype == 'Euclidean':
        dist = np.zeros((30,30))
        for i in np.arange(30):
            for j in np.arange(i):
                dist[i,j] = np.linalg.norm(points.iloc[:,i]-points.iloc[:,j])
        dist = dist + dist.T
    else:
        dist = np.zeros(30)
        
    pointer,levelset = myslink(dist)
    contained = np.where(levelset < epsilon)
    ptindex = [i for i in range(n)]
    result = []
    num = 0
    subsets = []
    while len(ptindex) != 0:
        tempcluster = []
        temp = ptindex.pop(0)
        while(1):
            tempcluster = tempcluster + [list(points.columns)[temp]]
            if not(temp in contained[0]):
                break
            temp = pointer[temp]
        subsets.append(tempcluster)
        
    tempind = list(set([c[-1] for c in subsets]))
    for i,endnum in zip(range(len(tempind)),tempind):
        cluster_ = []
        for c in subsets:
            if c[-1] == endnum:
                cluster_.extend(c)
                cluster_ = list(set(cluster_))
        result.append(cluster_)
    return result

def get_cluster_output(mds_f,data_features):
    r=0
    n_clust = np.inf
    clust = []
    clusters = {}
    limit = 0
    while(n_clust!=1 and limit<1000):
        mds_f_ = pd.DataFrame(mds_f.T,columns=data_features)
        temp = myclustering(mds_f_,0.002*r,'Euclidean')
        limit += 1
        r += 1
        if clust!=temp:
            n_clust = len(temp)
            clusters[n_clust] = temp
            clust = temp
    return clusters



def learning_model_test(x_train,y_train,x_test,y_test):
    '''
    Input
        x_train
        y_train
        x_test
        y_test
    Output
        acc_result : each model acc value
    
    '''
 
    print('\nCompare Multiple Classifiers: \n')
    print('K-Fold Cross-Validation Accuracy: \n')
    names = []
    models = []
    resultsAccuracy = []
    models.append(('LR', LogisticRegression()))
    models.append(('RF', RandomForestClassifier(n_estimators=100)))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('SVM', SVC()))
    models.append(('LSVM', LinearSVC()))
    models.append(('GNB', GaussianNB()))
    models.append(('DTC', DecisionTreeClassifier()))
    models.append(('GBC', GradientBoostingClassifier()))
    models.append(('XG', XGBClassifier()))
    
    for name, model in models:
        model.fit(x_train, y_train)
        kfold = KFold(n_splits=10, random_state=7)
        accuracy_results =cross_val_score(model, x_train,y_train, cv=kfold, scoring='accuracy')
        resultsAccuracy.append(accuracy_results)
        names.append(name)
        accuracyMessage = "%s: %f (%f)" % (name, accuracy_results.mean(), accuracy_results.std())
        print(accuracyMessage) 
    # Boxplot
    fig = plt.figure()
    fig.suptitle('Algorithm Comparison: Accuracy')
    ax = fig.add_subplot(111)
    plt.boxplot(resultsAccuracy)
    ax.set_xticklabels(names)
    ax.set_ylabel('Cross-Validation: Accuracy Score')
    plt.show()