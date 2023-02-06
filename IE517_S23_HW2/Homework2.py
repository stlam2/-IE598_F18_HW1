import pandas as pd
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

#Get data
Trea = pd.read_csv('C:\\Users\\dell\\Desktop\\IE 517\\Treasury Squeeze raw score data.csv')
x_Trea = Trea.loc[:,['price_crossing','price_distortion','roll_start','roll_heart','near_minus_next','ctd_last_first','ctd1_percent','delivery_cost','delivery_ratio']]
y_Trea = Trea.iloc[:,11]

#Prepare(Split)training and testing data
x_train, x_test, y_train, y_test = train_test_split(x_Trea, y_Trea,test_size=0.25, random_state=33)
print( x_train.shape, y_train.shape)

#KNN:
knn = KNeighborsClassifier(n_neighbors = 5)
scaler = StandardScaler()
x_train_std = scaler.fit_transform(x_train)
knn.fit(x_train_std,y_train)

#Testing for K
k_range = range (1,26)
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(x_train,y_train)
    y_pred = knn.predict(x_test)
    scores.append(accuracy_score(y_test,y_pred))
    
#Plot
x_axis = list(range(1,26))
plt.figure(dpi = 500)
plt.plot(x_axis,scores)
plt.xlabel("value of K")
plt.ylabel("Accuracy")

#Decision Tree:
#Fit the Tree
tree = DecisionTreeClassifier(criterion = 'gini',max_depth = 4,random_state = 1)
tree.fit(x_train,y_train)
y_pred_tree = tree.predict(x_test)
accuracy_result = accuracy_score(y_test,y_pred_tree)
print(accuracy_result)

#Visualize the tree
plt.figure(dpi = 500)
plot_tree(tree,filled = True)
plt.show()


print("My name ShunTat Lam")
print("My NetID is: stlam2")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")
