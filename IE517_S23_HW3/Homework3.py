import pandas as pd
import numpy as np
import pylab
import scipy.stats as stats
import matplotlib.pyplot as plt

#Get data
Data = pd.read_csv('C:\\Users\\dell\\Desktop\\IE 517\\IE517_S23_HW3\\HY_Universe_corporate bond.csv')

#EDA 01: Size:
count_row = Data.shape[0]  # Gives number of rows
count_col = Data.shape[1]  # Gives number of columns
print('The number of rows in the data frameis %i' %count_row )
print('The number of columns in the data frame is %i' %count_col)

#EDA 02: Nature of Attributes:
columns_cat = Data.applymap(np.isreal).all()
print('\n')
print('Are the columns numerical or not? (True for numerical)')
print(columns_cat)

#EDA 03: Statistical Summaries:

##03-01: empty values?
print('\n')
print('empty values in columns?')
print(Data[Data.isnull().T.any()]) 

##03-02: stat data  
#do stat only for numeric data 
non_floats = []
for col in Data:
    if Data[col].dtypes != "float64":
        non_floats.append(col)
Num_Data = Data.drop(columns=non_floats)
Average_data = np.average(Num_Data, axis=1)

##03-03 Correlation
total_mean_size = Num_Data['total_mean_size']
total_median_size = Num_Data['total_median_size']
pccs = np.corrcoef(total_mean_size,total_median_size)

#EDA 04: Visualization
##04_01:Q-Q plot
volumn_trade = Num_Data['volume_trades']
stats.probplot(volumn_trade, dist="norm", plot=pylab)
pylab.show()

##04_02: Interrelationship by scatter plot
### As an example, say I am curious about the relationship between volume_trades and total_mean_size

plt.figure(dpi = 500)
plt.scatter(volumn_trade,total_mean_size)
plt.xlabel("Volumn trade")
plt.ylabel("Total mean size")
plt.title("scatter plot between volumn trade and total mean size")


##04-04 Boxplot
weekly_mean_volume =  Num_Data['weekly_mean_volume']
weekly_median_volume = Num_Data['weekly_median_volume']
box_1,box_2 = weekly_mean_volume,weekly_median_volume
plt.figure(dpi = 500)
plt.title('trade amount boxplot')
plt.boxplot([box_1,box_2])
plt.show()

#Honorcode
print("My name ShunTat Lam")
print("My NetID is: stlam2")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")
