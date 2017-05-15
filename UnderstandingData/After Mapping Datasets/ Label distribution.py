import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn

Train_Data = pd.read_csv("../../PrepareData/ProducedData/Train_Data_categorized.csv",header=None)
Test_Data = pd.read_csv("../../PrepareData/ProducedData/Test_Data_categorized.csv",header=None)

labels_names = ['abnormal','normal']
labels = Train_Data.iloc[:][41]
labels_count = labels.value_counts()
percent = [(labels_count[1]/(labels_count[0]+labels_count[1])) , (labels_count[0]/(labels_count[0]+labels_count[1]))]

plt.bar(range(len(labels_count.index)),percent)
plt.xticks(range(len(labels_count.index)),labels_names,fontsize=15)
plt.title("label distribution (Train Dataset)")
plt.savefig("Label_distribution_Train.png")



labels_names = ['abnormal','normal']
labels = Test_Data.iloc[:][41]
labels_count = labels.value_counts()
percent = [(labels_count[1]/(labels_count[0]+labels_count[1])) , (labels_count[0]/(labels_count[0]+labels_count[1]))]

plt.bar(range(len(labels_count.index)),percent)
plt.xticks(range(len(labels_count.index)),labels_names,fontsize=15)
plt.title("label distribution (Test Dataset)")
plt.savefig("Label_distribution_Test.png")