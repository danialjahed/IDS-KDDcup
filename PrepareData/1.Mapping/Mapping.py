import pandas as pd
import numpy as np

Train_Data = pd.read_csv("../../DataSets/kddcup.data_10_percent_corrected.csv",header=None)
Test_Data = pd.read_csv("../../DataSets/corrected.csv",header=None)


# mapping String value to numeric value
for i in range(41):
    if type(Train_Data.iloc[0][i]) == str :

        values = Train_Data.iloc[:][i].unique()
        values = np.concatenate((values, Test_Data.iloc[:][i].unique()), axis=0)
        values = set(values)

        Dic = dict(zip(values, range(1,len(values)+1)))

        Train_Data.iloc[:][i] = Train_Data.iloc[:][i].map(Dic)
        Test_Data.iloc[:][i] = Test_Data.iloc[:][i].map(Dic)

#########################################

# mapping String value to numeric value for class label
# normal connections will be 0 and abnoraml connections will be 1
Dic = {"normal.":0}
Train_Data.iloc[:][41] = Train_Data.iloc[:][41].map(Dic)
Test_Data.iloc[:][41] = Test_Data.iloc[:][41].map(Dic)
Train_Data = Train_Data.fillna(1)
Test_Data = Test_Data.fillna(1)
#########################################


# print(Train_Data.isnull().values.any())
# print(Train_Data.isnull().sum())
# print(Test_Data.isnull().values.any())
# print(Test_Data.isnull().sum())

Train_Data.to_csv("../ProducedData/Train_Data_categorized.csv",index=False,header=False)
Test_Data.to_csv("../ProducedData/Test_Data_categorized.csv",index=False,header=False)
