import pandas as pd
import matplotlib.pyplot as plt
# import seaborn

Train_Data = pd.read_csv("../ProducedData/Train_Data_categorized.csv",header=None)
Test_Data = pd.read_csv("../ProducedData/Test_Data_categorized.csv",header=None)

# print(Train_Data.shape)

deleted_Features = []
l = len(Train_Data.iloc[0][:])
for i in range(l):
    if len(Train_Data.iloc[:][i].unique()) == 1:
        del Train_Data[i]
        del Test_Data[i]
        deleted_Features.append(i)


# print(Train_Data.shape)
pd.Series(deleted_Features).to_csv("deleted_Features.csv",index=False)


Train_Data.columns = range(len(Train_Data.iloc[0][:]))
Test_Data.columns = range(len(Test_Data.iloc[0][:]))

Correlation = Train_Data.corr()
FeatureSize = len(Train_Data.iloc[0][:])

# fig, ax = plt.subplots(figsize=(FeatureSize,FeatureSize))
# ax.matshow(Correlation)
# plt.xticks(range(len(Correlation.columns)),Correlation.columns,fontsize=50,rotation=90)
# plt.yticks(range(len(Correlation.columns)),Correlation.columns,fontsize=50)
# # plt.show()
# plt.savefig("Correlation_Map.png")


deleted_Features = []
for i in range(FeatureSize):
    for j in range(i+1,FeatureSize):
        if abs(Correlation[i][j]) > 0.999 :
            del Train_Data[i]
            del Test_Data[i]
            deleted_Features.append(i)
            break

# print(Train_Data.shape)

Train_Data.to_csv("../ProducedData/prepared_Train_Data_categorized.csv",index=False,header=False)
Test_Data.to_csv("../ProducedData/prepared_Test_Data_categorized.csv",index=False,header=False)

pd.Series(deleted_Features).to_csv("deleted_Features(with corr).csv",index=False)





