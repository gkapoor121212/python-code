import pandas as pd
file_data = pd.read_csv('Data.csv')
x = file_data.iloc[:,:-1].values
y = file_data.iloc[:,3].values

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis=0)
imputer = imputer.fit(x[:,1:3])
x[:,1:3] = imputer.transform(x[:,1:3])

from sklearn.preprocessing import LabelEncoder
label_encoder_x = LabelEncoder()
label_encoder_y = LabelEncoder()
x[:,0] = label_encoder_x.fit_transform(x[:,0])
y = label_encoder_y.fit_transform(y)
