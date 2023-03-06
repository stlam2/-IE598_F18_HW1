import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

#Get Data
Data = pd.read_csv('C:\\Users\\dell\\Desktop\\IE 517\\IE517_S23_HW6\\ccdefault.csv')

#Prepare(Split)training and testing data
x_values = Data.drop(columns = ["DEFAULT","ID"])
y_values = Data['DEFAULT']

#Part 1
for i in range(1,11):
    x_train, x_test, y_train, y_test = train_test_split(x_values, y_values,test_size=0.1, random_state= i)
    tree = DecisionTreeClassifier(criterion = 'gini',max_depth = 4,random_state = 1)
    tree.fit(x_train,y_train)
    accuracy_result_in = accuracy_score(y_train,tree.predict(x_train))
    accuracy_result_out = accuracy_score(y_test,tree.predict(x_test))
    print(f"When seed ={i}, the in sample accuracy is {accuracy_result_in}")
    print(f"When seed ={i}, the out of sample accuracy is {accuracy_result_out}")
    
#Part 2
tree = DecisionTreeClassifier(criterion = 'gini',max_depth = 4,random_state = 1)
cv_scores = cross_val_score(tree,x_values,y_values,cv = 10)
print("The cv scores:",cv_scores)


#Honorcode
print("My name ShunTat Lam")
print("My NetID is: stlam2")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")