import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn import linear_model
from sklearn import svm
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

#Get Data
Data = pd.read_csv('C:\\Users\\dell\\Desktop\\IE 517\\IE517_S23_HW5\\hw5_treasury yield curve data.csv')

#EDA
#EDA 01: Size:
count_row = Data.shape[0]  # Gives number of rows
count_col = Data.shape[1]  # Gives number of columns
print('The number of rows in the data frameis %i' %count_row )
print('The number of columns in the data frame is %i' %count_col)

##02: scattor plot
columns_plot = ['SVENF01','SVENF03','SVENF05','SVENF07','SVENF09']
sns.pairplot(Data[columns_plot],height = 2.5)
plt.show()

##03:heatmap
corr = np.corrcoef(Data[columns_plot].values.T)
sns.set(font_scale = 2)
heatmap = sns.heatmap(corr,cbar = True, annot = True,square = True, fmt = '.2f',annot_kws = {'size':15},yticklabels = columns_plot,xticklabels = columns_plot)
plt.show()

##03-01: empty values?
print('\n')
print('empty values in columns?')
print(Data[Data.isnull().T.any()]) 

#split data
x_values = Data.drop(columns = ["Adj_Close"])
x_values = x_values.drop(columns = ["Date"])
y_values = Data['Adj_Close']
x_train, x_test, y_train, y_test = train_test_split(x_values, y_values,test_size=0.15, random_state=42)

#Standardize
#data has been standardized

#Linear regression model
def train_and_evaluate(clf,x_train,y_train):
    clf.fit(x_train,y_train)
    print("Coefficient of determination on training set:", clf.score(x_train,y_train))
    cv = KFold(n_splits=5)
    scores = cross_val_score(clf,x_train,y_train,cv = cv)
    print("Average coefficient of determination using 5-fold crossvalidation:",np.mean(scores))

print("For linear regression before PCA")
clf_sgd = linear_model.SGDRegressor(loss = 'squared_loss',penalty = None, random_state = 42)
train_and_evaluate(clf_sgd,x_train,y_train)

print("For SVR regression before PCA")
clf_svr = svm.SVR(kernel = 'linear')
train_and_evaluate(clf_svr,x_train,y_train)

#PCA
pca = PCA(n_components=(30))
pca.fit(x_values)
evr1 = pca.explained_variance_ratio_
print("Explained variance ratio for all components:", evr1)

pca = PCA(n_components=(3))
pca.fit(x_values)
x_values_pca = pca.transform(x_values)
x_train_pca, x_test_pca = train_test_split(x_values_pca,test_size=0.15, random_state=42)
evr2 = pca.explained_variance_ratio_
print("Explained variance ratio for 3 components:", evr2)
cev = np.cumsum(evr2)
print("Cumulative explained variance for 3 components:", cev)

#fit after pca
print("For linear regression after PCA")
clf_sgd = linear_model.SGDRegressor(loss = 'squared_loss',penalty = None, random_state = 42)
train_and_evaluate(clf_sgd,x_train_pca,y_train)

print("For SVR regression after PCA")
clf_svr = svm.SVR(kernel = 'linear')
train_and_evaluate(clf_svr,x_train_pca,y_train)

#Honorcode
print("My name ShunTat Lam")
print("My NetID is: stlam2")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")
