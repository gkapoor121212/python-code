import pandas as pd
file_data = pd.read_csv('Data.csv')
x = file_data.iloc[:,:-1].values
y = file_data.iloc[:,3].values

from sklearn.cross_validation import train_test_split
X_train,X_Test,Y_Train,Y_Test = train_test_split(x,y,test_size=0.2,random_state=0)
