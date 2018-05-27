import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Dense, Flatten


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

Train_Data = Train_Data.as_matrix().reshape(Train_Data.shape[0], Train_Data.shape[1], 1)
Test_Data = Test_Data.as_matrix().reshape(Test_Data.shape[0], Test_Data.shape[1], 1)

Model = Sequential()
Model.add(Conv1D(20, 4, input_shape = Train_Data.shape[1:3] , activation = 'relu'))
Model.add(MaxPooling1D(2))
Model.add(Flatten())
Model.add(Dense(1, activation = 'sigmoid'))
sgd = SGD(lr = 0.1, momentum = 0.9, decay = 0, nesterov = False)
Model.compile(loss = 'binary_crossentropy', optimizer = sgd, metrics = ['accuracy'])
Model.fit(Train_Data, Train_Data_labels, batch_size = 500, epochs = 1, verbose = 1)

print("Result on PCA with 25 components :")
Predicted_Train = Model.predict(Train_Data)
for i in range(len(Predicted_Train)):
    if Predicted_Train[i] <= 0.5:
        Predicted_Train[i] = 0
    else:
        Predicted_Train[i] = 1
print("Accuracy on Training Data :",metrics.accuracy_score(Train_Data_labels,Predicted_Train))
Predicted_Train = Model.predict(Test_Data)
for i in range(len(Predicted_Train)):
    if Predicted_Train[i] <= 0.5:
        Predicted_Train[i] = 0
    else:
        Predicted_Train[i] = 1
print("Accuracy on Testing Data :",metrics.accuracy_score(Test_Data_labels,Predicted_Train))
print("\nconfusion_matrix : \n",metrics.confusion_matrix(Test_Data_labels,Predicted_Train,labels=[0,1]))
print("\nclassification report : \n\n",metrics.classification_report(Test_Data_labels,Predicted_Train,labels=[0,1]))
print("**********************************************************************")
exit()
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

conv = Sequential()
conv.add(Conv1D(20, 4, input_shape = x_train.shape[1:3], activation = 'relu'))
conv.add(MaxPooling1D(2))
conv.add(Flatten())
conv.add(Dense(1, activation = 'sigmoid'))
sgd = SGD(lr = 0.1, momentum = 0.9, decay = 0, nesterov = False)
conv.compile(loss = 'binary_crossentropy', optimizer = sgd, metrics = ['accuracy'])
conv.fit(x_train, y_train, batch_size = 500, epochs = 100, verbose = 0)

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

conv = Sequential()
conv.add(Conv1D(20, 4, input_shape = x_train.shape[1:3], activation = 'relu'))
conv.add(MaxPooling1D(2))
conv.add(Flatten())
conv.add(Dense(1, activation = 'sigmoid'))
sgd = SGD(lr = 0.1, momentum = 0.9, decay = 0, nesterov = False)
conv.compile(loss = 'binary_crossentropy', optimizer = sgd, metrics = ['accuracy'])
conv.fit(x_train, y_train, batch_size = 500, epochs = 100, verbose = 0)

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
plt.title('Naive bayes result')
plt.xticks(index + bar_width, ('PCA with 25 cmpnt', 'PCA with full cmpnt', 'Without PCA'))
plt.legend()

plt.tight_layout()
plt.savefig("Naive_bayes_result.png")