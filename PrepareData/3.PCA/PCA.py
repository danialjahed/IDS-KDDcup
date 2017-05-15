import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as PCA
import matplotlib.pyplot as plt
import seaborn

Train_Data = pd.read_csv("../ProducedData/prepared_Train_Data_categorized.csv",header=None)
Test_Data = pd.read_csv("../ProducedData/prepared_Test_Data_categorized.csv",header=None)

Labels_index = len(Train_Data.iloc[0][:])-1
Train_Data_labels = Train_Data.iloc[:][Labels_index]
del Train_Data[Labels_index]
Test_Data_labels = Test_Data.iloc[:][Labels_index]
del Test_Data[Labels_index]


Standarder = StandardScaler().fit(Train_Data)
Train_Data_std = Standarder.transform(Train_Data)
Test_Data_std = Standarder.transform(Test_Data)


pca = PCA(n_components=25).fit(Train_Data_std)
PCA_Train_Data = pca.transform(Train_Data_std)
PCA_Test_Data = pca.transform(Test_Data_std)

df = pd.DataFrame(PCA_Train_Data)
df[25] = Train_Data_labels
df.to_csv("../ProducedData/PCA(25)_prepared_Train_Data.csv",index=False,header=False)

df = pd.DataFrame(PCA_Test_Data)
df[25] = Test_Data_labels
df.to_csv("../ProducedData/PCA(25)_prepared_Test_Data.csv",index=False,header=False)


pca = PCA().fit(Train_Data_std)
PCA_Train_Data = pca.transform(Train_Data_std)
PCA_Test_Data = pca.transform(Test_Data_std)

df = pd.DataFrame(PCA_Train_Data)
df[Labels_index] = Train_Data_labels
df.to_csv("../ProducedData/PCA(full)_prepared_Train_Data.csv",index=False,header=False)

df = pd.DataFrame(PCA_Test_Data)
df[Labels_index] = Test_Data_labels
df.to_csv("../ProducedData/PCA(full)_prepared_Test_Data.csv",index=False,header=False)


# plt.plot(np.cumsum(pca.explained_variance_ratio_))
# plt.xlabel('number of components')
# plt.ylabel('cumulative explained variance')
# plt.savefig("PCA(n_components).png")
