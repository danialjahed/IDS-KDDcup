import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics


accuracy = []
precision = []
recall = []
f1 = []


Train_Data = pd.read_csv("../PrepareData/ProducedData/PCA(25)_prepared_Train_Data.csv",header=None)
Test_Data = pd.read_csv("../PrepareData/ProducedData/PCA(25)_prepared_Test_Data.csv",header=None)

Labels_index = len(Train_Data.iloc[0][:])-1
Train_Data_labels = Train_Data.iloc[:][Labels_index]
del Train_Data[Labels_index]
Test_Data_labels = Test_Data.iloc[:][Labels_index]
del Test_Data[Labels_index]

Model = RandomForestClassifier(random_state=30).fit(Train_Data,Train_Data_labels)

print("Result on PCA with 25 components :")
Predicted_Train = Model.predict(Train_Data)
print("Accuracy on Training Data :",metrics.accuracy_score(Train_Data_labels,Predicted_Train))

Predicted_Train = Model.predict(Test_Data)
print("Accuracy on Testing Data :",metrics.accuracy_score(Test_Data_labels,Predicted_Train))
print("\nconfusion_matrix : \n",metrics.confusion_matrix(Test_Data_labels,Predicted_Train,labels=[0,1]))
print("\nclassification report : \n\n",metrics.classification_report(Test_Data_labels,Predicted_Train,labels=[0,1]))
print("**********************************************************************")

accuracy.append(metrics.accuracy_score(Test_Data_labels,Predicted_Train))
precision.append(metrics.precision_score(Test_Data_labels,Predicted_Train))
recall.append(metrics.recall_score(Test_Data_labels,Predicted_Train))
f1.append(metrics.f1_score(Test_Data_labels,Predicted_Train,labels=[0,1]))


Train_Data = pd.read_csv("../PrepareData/ProducedData/PCA(full)_prepared_Train_Data.csv",header=None)
Test_Data = pd.read_csv("../PrepareData/ProducedData/PCA(full)_prepared_Test_Data.csv",header=None)

Labels_index = len(Train_Data.iloc[0][:])-1
Train_Data_labels = Train_Data.iloc[:][Labels_index]
del Train_Data[Labels_index]
Test_Data_labels = Test_Data.iloc[:][Labels_index]
del Test_Data[Labels_index]

Model = RandomForestClassifier(random_state=30).fit(Train_Data,Train_Data_labels)

print("Result on PCA with full components :")
Predicted_Train = Model.predict(Train_Data)
print("Accuracy on Training Data :",metrics.accuracy_score(Train_Data_labels,Predicted_Train))

Predicted_Train = Model.predict(Test_Data)
print("Accuracy on Testing Data :",metrics.accuracy_score(Test_Data_labels,Predicted_Train))
print("\nconfusion_matrix : \n",metrics.confusion_matrix(Test_Data_labels,Predicted_Train,labels=[0,1]))
print("\nclassification report : \n\n",metrics.classification_report(Test_Data_labels,Predicted_Train,labels=[0,1]))
print("**********************************************************************")

accuracy.append(metrics.accuracy_score(Test_Data_labels,Predicted_Train))
precision.append(metrics.precision_score(Test_Data_labels,Predicted_Train))
recall.append(metrics.recall_score(Test_Data_labels,Predicted_Train))
f1.append(metrics.f1_score(Test_Data_labels,Predicted_Train,labels=[0,1]))

Train_Data = pd.read_csv("../PrepareData/ProducedData/prepared_Train_Data_categorized.csv",header=None)
Test_Data = pd.read_csv("../PrepareData/ProducedData/prepared_Test_Data_categorized.csv",header=None)


Labels_index = len(Train_Data.iloc[0][:])-1
Train_Data_labels = Train_Data.iloc[:][Labels_index]
del Train_Data[Labels_index]
Test_Data_labels = Test_Data.iloc[:][Labels_index]
del Test_Data[Labels_index]

Model = RandomForestClassifier(random_state=30).fit(Train_Data,Train_Data_labels)

print("Result on PCA with 25 components")
Predicted_Train = Model.predict(Train_Data)
print("Accuracy on Training Data :",metrics.accuracy_score(Train_Data_labels,Predicted_Train))

Predicted_Train = Model.predict(Test_Data)
print("Accuracy on Testing Data :",metrics.accuracy_score(Test_Data_labels,Predicted_Train))
print("\nconfusion_matrix : \n",metrics.confusion_matrix(Test_Data_labels,Predicted_Train,labels=[0,1]))
print("\nclassification report : \n\n",metrics.classification_report(Test_Data_labels,Predicted_Train,labels=[0,1]))
print("**********************************************************************")

accuracy.append(metrics.accuracy_score(Test_Data_labels,Predicted_Train))
precision.append(metrics.precision_score(Test_Data_labels,Predicted_Train))
recall.append(metrics.recall_score(Test_Data_labels,Predicted_Train))
f1.append(metrics.f1_score(Test_Data_labels,Predicted_Train,labels=[0,1]))



n_groups = 3
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.1
opacity = 0.8

rects1 = plt.bar(index, accuracy, bar_width,
                 alpha=opacity,
                 color='b',
                 label='accuracy')

rects2 = plt.bar(index + bar_width, precision, bar_width,
                 alpha=opacity,
                 color='g',
                 label='precision')

rects3 = plt.bar(index + 2*bar_width, recall, bar_width,
                 alpha=opacity,
                 color='r',
                 label='recall')

rects4 = plt.bar(index + 3*bar_width, f1, bar_width,
                 alpha=opacity,
                 color='y',
                 label='f1')

plt.xlabel('Datasets')
plt.ylabel('Scores')
plt.title('Random Forest Result(state = 30)')
plt.xticks(index + bar_width, ('PCA with 25 cmpnt', 'PCA with full cmpnt', 'Without PCA'))
plt.legend()

plt.tight_layout()

plt.savefig("Random_Forest_Result.png")