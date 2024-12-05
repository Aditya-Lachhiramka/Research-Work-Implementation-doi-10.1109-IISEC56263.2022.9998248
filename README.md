# Research-Work-Implementation-doi-10.1109/IISEC56263.2022.9998248

Performance Comparison of Different Machine Learning Techniques for Early Prediction of Breast Cancer using Wisconsin Breast Cancer Dataset
1. Introduction: The detection of breast cancer at an early stage is critical for improving treatment outcomes. In this report, we compare several machine learning models to evaluate their performance in predicting the diagnosis of breast cancer. The Wisconsin Breast Cancer Dataset, which contains various features related to cell measurements, was used for this purpose.
2. Data Preprocessing: The dataset was loaded from a CSV file and underwent cleaning by removing unnecessary columns such as id and Unnamed: 32. Additionally, the target variable diagnosis, which was originally categorical (M for malignant, B for benign), was mapped to numerical values (1 for malignant and 0 for benign). Low-correlation features were dropped, and missing values were handled by removing rows with NaN values. The features were then standardized using StandardScaler, and the data was split into training and testing sets.
3. Models Evaluated: The following machine learning and deep learning models were trained and evaluated:
•	K-Nearest Neighbours (KNN)
•	Naive Bayes (NB)
•	Support Vector Machine (SVM)
•	Decision Tree (DT)
•	Logistic Regression (LR)
•	Deep Learning (DL)
•	Artificial Neural Network (ANN)
•	Multi-Layer Perceptron (MLP)
•	Random Forest (RF)
•	AdaBoost
•	Deep Forest (DF)
•	Deep Neural Network (DNN)
•	Softmax Regression
•	Gradient Boost Decision Tree (GBDT)
4. Model Performance: Each model was evaluated using accuracy, which is the proportion of correct predictions. The performance metrics were extracted from classification reports generated after training each model on the training set and testing it on the test set.
5. Results: The accuracy of each model was compiled and sorted to identify the best-performing model:
Model	Accuracy
Artificial Neural Network(ANN)	0.99
Random Forest	0.98
Gradient Boost Decision Tree (GBDT)	0.98
Deep Neural Network (DNN)	0.97
Multi-Layer Perceptron (MLP)	0.96
Logistic Regression (LR)	0.96
Support Vector Machine (SVM)	0.95
AdaBoost	0.94
Deep Forest (DF)	0.94
K-Nearest Neighbours (KNN)	0.93
Decision Tree (DT)	0.91
Softmax Regression	0.89
Naive Bayes (NB)	0.88
6. Discussion:
•	ANN achieved the highest accuracy, making it the best model for this task, closely followed by Random Forest and Gradient Boost Decision Tree (GBDT).
•	Traditional models like Logistic Regression, KNN, and SVM performed well but were outperformed by ensemble methods and deep learning techniques.
•	Naive Bayes and Softmax Regression showed lower performance, highlighting the benefit of more complex models for this dataset.
7. Conclusion: Based on the accuracy metric, the Artificial Neural Network (ANN) is the best-performing model for predicting breast cancer using the Wisconsin Breast Cancer Dataset. Ensemble methods such as Random Forest and Gradient Boosting Decision Tree (GBDT) also show strong performance and could serve as robust alternatives. While simpler models like Logistic Regression and Naive Bayes are faster and less computationally intensive, they do not achieve the same level of predictive accuracy as the more advanced models.
8. Future Work: For future improvements, hyperparameter tuning and cross-validation could further enhance the models' performance. Additionally, exploring other features or incorporating additional datasets may yield better results for early-stage breast cancer prediction.





