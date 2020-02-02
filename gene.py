import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

import sklearn
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.preprocessing import StandardScaler
from sklearn import  linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm

from sklearn.neural_network import MLPClassifier
# from random import randint
from sklearn.model_selection import train_test_split


def preprocess(path):
    # preprocess

    df = pd.read_csv(path)

    df1 = [col for col in df.columns if "call" not in col]
    df = df[df1]

    df = df.T
    df2 = df.drop(['Gene Description', 'Gene Accession Number'], axis=0)

    df2.index = pd.to_numeric(df2.index)
    df2.sort_index(inplace=True)

    return df2





# pca dimension reduction
def pca(df2):
    X_std = StandardScaler().fit_transform(df2.drop('cat', axis=1))

    sklearn_pca = sklearnPCA(n_components=30)
    Y_sklearn = sklearn_pca.fit_transform(X_std)

    cum_sum = sklearn_pca.explained_variance_ratio_.cumsum()

    sklearn_pca.explained_variance_ratio_[:10].sum()

    # cum_sum = cum_sum * 100

    # fig, ax = plt.subplots(figsize=(8, 8))
    # plt.bar(range(30), cum_sum, label='Cumulative _Sum_of_Explained _Varaince', color='b', alpha=0.5)
    # plt.title("Around 95% of variance is explained by the Fisrt 30 colmns ")
    # plt.show()

    X_reduced2 = Y_sklearn

    train = pd.DataFrame(X_reduced2)

    train['cat'] = df2['cat'].reset_index().cat


    #  show the effect of pca:


    # sklearn_pca = sklearnPCA(n_components=3)
    # X_reduced = sklearn_pca.fit_transform(X_std)
    # Y = train['cat']
    # from mpl_toolkits.mplot3d import Axes3D
    # plt.clf()
    # # fig = plt.figure(1, figsize=(10,6 ))
    # # ax = Axes3D(fig, elev=-150, azim=110,)
    # # ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=Y,cmap=plt.cm.Paired,linewidths=10)
    # # ax.set_title("First three PCA directions")
    # # ax.set_xlabel("1st eigenvector")
    # # ax.w_xaxis.set_ticklabels([])
    # # ax.set_ylabel("2nd eigenvector")
    # # ax.w_yaxis.set_ticklabels([])
    # # ax.set_zlabel("3rd eigenvector")
    # # ax.w_zaxis.set_ticklabels([])
    #
    # # plt.show()
    #
    # # import matplotlib.pyplot as plt
    # fig = plt.figure(1, figsize=(10, 6))
    # plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=df2['cat'], cmap=plt.cm.Paired, linewidths=10)
    # plt.annotate('See The Brown Cluster', xy=(20, -20), xytext=(9, 8), arrowprops=dict(facecolor='black', shrink=0.05))
    # # plt.scatter(test_reduced[:, 0],  test_reduced[:, 1],c='r')
    # plt.title("This The 2D Transformation of above graph ")
    # plt.show()
    return X_reduced2,train


# classification models
# 1.knn
def knn(train,test):
    clf = KNeighborsClassifier(n_neighbors=10, )
    clf.fit(train.drop('cat', axis=1), train['cat'])
    pred = clf.predict(test.drop('cat', axis=1))
    true=test['cat']
    print('knn:\n',sklearn.metrics.confusion_matrix(true, pred))

# 2.lr
def lr(train,test):

    clf = linear_model.LinearRegression()

    clf.fit(train.drop('cat', axis=1), train['cat'])
    pred = clf.predict(test.drop('cat', axis=1))
    true = test['cat']
    # print(true)
    # print(pred)

    pred=[1  if abs(i-1)>abs(i-0) else 0 for i in pred]
    print('lr:\n',sklearn.metrics.confusion_matrix(true, pred))


# 3.naive bayes
def nb(train,test):

    clf=GaussianNB()

    clf.fit(train.drop('cat', axis=1), train['cat'])
    pred = clf.predict(test.drop('cat', axis=1))
    true = test['cat']
    print('nb:\n',sklearn.metrics.confusion_matrix(true, pred))

# 4.decision tree

def decisionTree(train,test):
    clf = DecisionTreeClassifier(min_samples_split=2)
    clf.fit(train.drop('cat', axis=1), train['cat'])
    pred = clf.predict(test.drop('cat', axis=1))
    true = test['cat']
    print('decisionTree:\n',sklearn.metrics.confusion_matrix(true, pred))

# 5.svm
def svm_train(train,test):

    clf = svm.SVC(kernel='linear')
    clf.fit(train.drop('cat', axis=1), train['cat'])
    pred = clf.predict(test.drop('cat', axis=1))
    true = test['cat']
    print('svm:\n',sklearn.metrics.confusion_matrix(true, pred))


# 6.neural network
def nn(train,test):

    clf=MLPClassifier(solver='lbfgs',alpha=1e-5,hidden_layer_sizes=(5,2),random_state=1)

    # clf=MLPClassifier(solver = 'lbfgs', alpha = 1e-5, hidden_layer_sizes = (5, 2), random_state = 1, learning_rate_init = 0.05)

    clf.fit(train.drop('cat', axis=1), train['cat'])
    pred = clf.predict(test.drop('cat', axis=1))
    true = test['cat']
    confusionMatrics=sklearn.metrics.confusion_matrix(true, pred)
    print('nn:\n',confusionMatrics)
    return confusionMatrics




# plot train and test dataset
def visualization(train,test,train_reduced,test_reduced):

    plt.clf()
    fig = plt.figure(1, figsize=(14, 6))

    plt.scatter(train_reduced[ 0], train_reduced[1], c=train_reduced['cat'], cmap=plt.cm.Paired, alpha=1, linewidths=7)
    # plt.scatter(train[:, 0], train[:, 1], c=train_reduced['cat'], cmap=plt.cm.Paired, alpha=0.7, linewidths=7)
    plt.scatter(test_reduced[0], test_reduced[1], c=test_reduced['cat'], cmap=plt.cm.Paired, alpha=0.5,linewidths=7)
    plt.show()


# cross validation
def crossValidation(train,test):


    dataset=pd.concat([train, test], ignore_index=True)

    X_train, X_test, y_train, y_test = train_test_split(dataset, dataset['cat'],test_size=0.15)

    # print(len(X_train), len(X_test))

    return X_train,X_test






def main():
    df2=preprocess('data/data_set_ALL_AML_train.csv')


    df2['cat'] = list(pd.read_csv('data/actual.csv')[:38]['cancer'])
    dic = {'ALL': 0, 'AML': 1}
    df2.replace(dic, inplace=True)
    train_view,train_reduced=pca(df2)


    test2=preprocess('data/data_set_ALL_AML_independent.csv')
    test2['cat'] = list(pd.read_csv('data/actual.csv')[38:]['cancer'])
    dic = {'ALL': 0, 'AML': 1}
    test2.replace(dic, inplace=True)

    # use pca to process the dataset
    test_view,test_reduced=pca(test2)


    # use cross validation to make the best use of dataset
    for i in range(10):

        new_train,new_test=crossValidation(train_reduced,test_reduced)
        confusionMatrics=nn(new_train, new_test)
        accuracy=confusionMatrics[0][0]+confusionMatrics[1][1]
        print(accuracy)


    # different kinds of classification models, used in cross validation loop

    # knn(train_reduced,test_reduced)
    # nb(train_reduced,test_reduced)
    # lr(train_reduced,test_reduced)
    #
    # decisionTree(train_reduced,test_reduced)
    # svm_train(train_reduced,test_reduced)
    # nn(train_reduced,test_reduced)



    # plot the result
    visualization(train_view,test_view,train_reduced,test_reduced)


if __name__ == '__main__':


    main()


