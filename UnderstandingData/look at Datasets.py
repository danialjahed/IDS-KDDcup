import pandas as pd
import matplotlib.pyplot as plt
import seaborn


#Load train dataset
Train_Data = pd.read_csv("../DataSets/kddcup.data_10_percent_corrected.csv",header=None)
print("train :",Train_Data.shape)

#visulize data variety
labels = Train_Data.iloc[:][41]
labels_count = labels.value_counts()
plt.bar(range(len(labels_count.index)),labels_count.values)
plt.xticks(range(len(labels_count.index)),labels_count.index,fontsize=12,rotation=90)
plt.savefig("labels_variety(for Training Dataset).png")


#checking missing values of Traindata
print(Train_Data.isnull().values.any())
# print(Train_Data.isnull().sum())


#Load test dataset
Test_Data = pd.read_csv("../DataSets/corrected.csv",header=None)
print("test :", Test_Data.shape)

#checking missing values of Testdata
print(Test_Data.isnull().values.any())
# print(Test_Data.isnull().sum())

#visulize data variety
labels = Test_Data.iloc[:][41]
labels_count = labels.value_counts()
plt.bar(range(len(labels_count.index)),labels_count.values)
plt.xticks(range(len(labels_count.index)),labels_count.index,fontsize=8,rotation=90)
plt.savefig("labels_variety(for Testing Dataset).png")