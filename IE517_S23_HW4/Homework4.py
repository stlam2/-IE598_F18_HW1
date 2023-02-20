import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error

#Get Data
Data = pd.read_csv('C:\\Users\\dell\\Desktop\\IE 517\\IE517_S23_HW4\\housing.csv')

#EDA
##01: scattor plot
columns_plot = ['LSTAT','INDUS','NOX','RM','MEDV']
sns.pairplot(Data[columns_plot],height = 2.5)
plt.show()

##02:heatmap
corr = np.corrcoef(Data[columns_plot].values.T)
sns.set(font_scale = 2)
heatmap = sns.heatmap(corr,cbar = True, annot = True,square = True, fmt = '.2f',annot_kws = {'size':15},yticklabels = columns_plot,xticklabels = columns_plot)
plt.show()

#03:data split
x_house = Data.drop(columns = ["MEDV"])
y_house = Data['MEDV']
x_train, x_test, y_train, y_test = train_test_split(x_house, y_house,test_size=0.2, random_state=42)

#04: Linear regression
sc_x = StandardScaler()
sc_y = StandardScaler()
x_std = sc_x.fit_transform(x_train)
y_std = sc_y.fit_transform(y_train[:,np.newaxis]).flatten()
lr = LinearRegression()
lr.fit(x_std,y_std)

##coefficients and intercet
print('Slope 1 :%.3f'% lr.coef_[0])
print('Slope 2 :%.3f'% lr.coef_[1])
print('Slope 3 :%.3f'% lr.coef_[2])
print('Slope 4 :%.3f'% lr.coef_[3])
print('Slope 5 :%.3f'% lr.coef_[4])
print('Slope 6 :%.3f'% lr.coef_[5])
print('Slope 7 :%.3f'% lr.coef_[6])
print('Slope 8 :%.3f'% lr.coef_[7])
print('Slope 9 :%.3f'% lr.coef_[8])
print('Slope 10 :%.3f'% lr.coef_[9])
print('Slope 11 :%.3f'% lr.coef_[10])
print('Slope 12 :%.3f'% lr.coef_[11])
print('Slope 13 :%.3f'% lr.coef_[12])
print('Intercept:%.3f'% lr.intercept_)

##Residual
y_train_pred = lr.predict(x_train)
y_test_pred = lr.predict(x_test)
plt.scatter(y_train_pred,y_train_pred - y_train, c = 'steelblue',marker = 'o',edgecolor = 'white',label = 'Training data')
plt.scatter(y_test_pred,y_test_pred - y_test, c = 'limegreen',marker = 's',edgecolor = 'white',label = 'Test data')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.legend(loc = 'upper left')
plt.hlines(y = 0,xmin = -10, xmax = 50, color = 'black', lw = 2)
plt.xlim = ([-10,50])
plt.show()

## MSE
print('MSE train: %.3f, test:%.3f'%(mean_squared_error(y_train,y_train_pred),mean_squared_error(y_test,y_test_pred)))

##R_squared
SSR = sum((y_train - y_train_pred)**2)
SST = sum((y_train - np.mean(y_train))**2)
r_squared = 1 - (float(SSR))/SST
adjusted_r_squared = 1 - (1 - r_squared)*(len(y_train)-1)/(len(y_train)-x_train.shape[1]-1)
print('R square :%.3f'% r_squared)

#Ridge Regression:
n_alphas = 200
alpha= np.logspace(0,1,n_alphas)
coefs = []
for a in alpha:
    ridge = Ridge(alpha=a)
    ridge.fit(x_std, y_std)
    coefs.append(ridge.coef_)

ax = plt.gca()
ax.plot(alpha, coefs)
ax.set_xscale("log")
ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
plt.xlabel("alpha")
plt.ylabel("weights")
plt.title("Ridge coefficients as a function of the Ridge regularization")
plt.axis("tight")
plt.show()
#alpha does not affact heavily
ridge = Ridge(alpha=100)
ridge.fit(x_train,y_train)
##coefficients and intercet
print('Slope 1 :%.3f'% ridge.coef_[0])
print('Slope 2 :%.3f'% ridge.coef_[1])
print('Slope 3 :%.3f'% ridge.coef_[2])
print('Slope 4 :%.3f'% ridge.coef_[3])
print('Slope 5 :%.3f'% ridge.coef_[4])
print('Slope 6 :%.3f'% ridge.coef_[5])
print('Slope 7 :%.3f'% ridge.coef_[6])
print('Slope 8 :%.3f'% ridge.coef_[7])
print('Slope 9 :%.3f'% ridge.coef_[8])
print('Slope 10 :%.3f'% ridge.coef_[9])
print('Slope 11 :%.3f'% ridge.coef_[10])
print('Slope 12 :%.3f'% ridge.coef_[11])
print('Slope 13 :%.3f'% ridge.coef_[12])
print('Intercept:%.3f'% ridge.intercept_)
##Residual
y_train_pred = ridge.predict(x_train)
y_test_pred = ridge.predict(x_test)
plt.scatter(y_train_pred,y_train_pred - y_train, c = 'steelblue',marker = 'o',edgecolor = 'white',label = 'Training data')
plt.scatter(y_test_pred,y_test_pred - y_test, c = 'limegreen',marker = 's',edgecolor = 'white',label = 'Test data')
plt.xlabel('Predicted Values (Ridge)')
plt.ylabel('Residuals')
plt.legend(loc = 'upper left')
plt.hlines(y = 0,xmin = -10, xmax = 50, color = 'black', lw = 2)
plt.xlim = ([-10,50])
plt.show()
## MSE
print('MSE train: %.3f, test:%.3f'%(mean_squared_error(y_train,y_train_pred),mean_squared_error(y_test,y_test_pred)))

##R_squared
SSR = sum((y_train - y_train_pred)**2)
SST = sum((y_train - np.mean(y_train))**2)
r_squared = 1 - (float(SSR))/SST
adjusted_r_squared = 1 - (1 - r_squared)*(len(y_train)-1)/(len(y_train)-x_train.shape[1]-1)
print('R square :%.3f'% r_squared)

#Lasso Regression:
alphas = np.linspace(0.01,500,100)
lasso = Lasso(max_iter=10000)
coefs = []

for a in alphas:
    lasso.set_params(alpha=a)
    lasso.fit(x_train, y_train)
    coefs.append(lasso.coef_)

ax = plt.gca()

ax.plot(alphas, coefs)
ax.set_xscale('log')
plt.axis('tight')
plt.xlabel('alpha')
plt.ylabel('Standardized Coefficients')
plt.title('Lasso coefficients as a function of alpha');

lasso = Lasso(alpha = -100)

lasso.set_params(alpha=1)
lasso.fit(x_std, y_std)
##coefficients and intercet
print('Slope 1 :%.3f'% lasso.coef_[0])
print('Slope 2 :%.3f'% lasso.coef_[1])
print('Slope 3 :%.3f'% lasso.coef_[2])
print('Slope 4 :%.3f'% lasso.coef_[3])
print('Slope 5 :%.3f'% lasso.coef_[4])
print('Slope 6 :%.3f'% lasso.coef_[5])
print('Slope 7 :%.3f'% lasso.coef_[6])
print('Slope 8 :%.3f'% lasso.coef_[7])
print('Slope 9 :%.3f'% lasso.coef_[8])
print('Slope 10 :%.3f'% lasso.coef_[9])
print('Slope 11 :%.3f'% lasso.coef_[10])
print('Slope 12 :%.3f'% lasso.coef_[11])
print('Slope 13 :%.3f'% lasso.coef_[12])
print('Intercept:%.3f'% lasso.intercept_)
##Residual
#y_train_pred = lasso.predict(x_train)
#y_test_pred = lasso.predict(x_test)
#plt.scatter(y_train_pred,y_train_pred - y_train, c = 'steelblue',marker = 'o',edgecolor = 'white',label = 'Training data')
#plt.scatter(y_test_pred,y_test_pred - y_test, c = 'limegreen',marker = 's',edgecolor = 'white',label = 'Test data')
#plt.xlabel('Predicted Values (Lasso)')
#plt.ylabel('Residuals')
#plt.legend(loc = 'upper left')
#plt.hlines(y = 0,xmin = -10, xmax = 50, color = 'black', lw = 2)
#plt.xlim = ([-10,50])
#plt.show()
## MSE
print('MSE train: %.3f, test:%.3f'%(mean_squared_error(y_train,y_train_pred),mean_squared_error(y_test,y_test_pred)))

##R_squared
SSR = sum((y_train - y_train_pred)**2)
SST = sum((y_train - np.mean(y_train))**2)
r_squared = 1 - (float(SSR))/SST
adjusted_r_squared = 1 - (1 - r_squared)*(len(y_train)-1)/(len(y_train)-x_train.shape[1]-1)
print('R square :%.3f'% r_squared)

#Honorcode
print("My name ShunTat Lam")
print("My NetID is: stlam2")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")