# IDS-KDDcup
Detecting Abnormal Connections of a Network Flow(KDD-cup 99)

## Understanding Data
try to find out what is dataset different classes

![alt tag](UnderstandingData/labels_variety(for%20Trainig%20Dataset).png)
![alt tag](UnderstandingData/labels_variety(for%20Testing%20Dataset).png)

### after mapping string value to numeric and divide all class to normal and abnormal
![alt tag](UnderstandingData/After%20Mapping%20Datasets/Label_distribution_Train.png)

![alt tag](UnderstandingData/After%20Mapping%20Datasets/Label_distribution_Test.png)

## Prepare Data
  ### Feature redunction
![alt tag](PrepareData/2.FeatureReduction/Correlation_Map.png)

![alt tag](PrepareData/2.FeatureReduction/Correlation_Map(seaborn).png)

  ### PCA
![alt tag](PrepareData/3.PCA/PCA(n_components=10).png)

![alt tag](PrepareData/3.PCA/PCA(n_components=25).png)

![alt tag](PrepareData/3.PCA/PCA(n_components=35).png)

![alt tag](PrepareData/3.PCA/PCA(n_components%20Compelete).png)


## IDS
   ### Naive bayes
![alt tag](IDS/Naive_bayes_result.png)
  ### Random Forest
![alt tag](IDS/Random_Forest_Result.png)
  ### Logistic Regression
![alt tag](IDS/Logistic_Regression_Result.png)
  ### Decision Tree
![alt tag](IDS/Decision_Tree_result.png)
  ### SVC
![alt tag](IDS/SVC_result.png)
  ### Compare algorithms
![alt tag](IDS/Algorithms%20diffrence%20on%20f1.png)

![alt tag](IDS/Algorithms%20results.png)
