
import pandas as pd

file_path = 'data.csv'
data = pd.read_csv(file_path)

data_shape = data.shape
data_head = data.head()

data_shape, data_head

import matplotlib.pyplot as plt
import seaborn as sns

data_cleaned = data.drop(columns=['id', 'Unnamed: 32'])

data_cleaned['diagnosis'] = data_cleaned['diagnosis'].map({'M': 1, 'B': 0})

correlation_matrix = data_cleaned.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', cbar=True)
plt.title("Correlation Matrix Heatmap")
plt.show()

target_correlation = correlation_matrix['diagnosis'].sort_values(ascending=False)
target_correlation.head(10), target_correlation.tail(10)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

low_corr_features = ['smoothness_se', 'symmetry_se', 'texture_se', 'fractal_dimension_mean']
data_preprocessed = data_cleaned.drop(columns=low_corr_features)

missing_values = data_preprocessed.isnull().sum()

data_preprocessed = data_preprocessed.dropna()

X = data_preprocessed.drop(columns=['diagnosis'])
y = data_preprocessed['diagnosis']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

X_train.shape, X_test.shape, y_train.shape, y_test.shape

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

knn_report = classification_report(y_test, y_pred_knn, output_dict=False)

print(knn_report)

from sklearn.naive_bayes import GaussianNB
nb_classifier = GaussianNB()

nb_classifier.fit(X_train, y_train)
y_pred_nb = nb_classifier.predict(X_test)

nb_report = classification_report(y_test, y_pred_nb, output_dict= False)

print(nb_report)

from sklearn.svm import SVC

svm_classifier = SVC(kernel='linear', random_state=42)
svm_classifier.fit(X_train, y_train)

y_pred_svm = svm_classifier.predict(X_test)

svm_report = classification_report(y_test, y_pred_svm, output_dict=False)

print(svm_report)

from sklearn.tree import DecisionTreeClassifier

dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)

y_pred_dt = dt_classifier.predict(X_test)

dt_report = classification_report(y_test, y_pred_dt, output_dict=False)

print(dt_report)

from sklearn.linear_model import LogisticRegression

lr_classifier = LogisticRegression(random_state=42)
lr_classifier.fit(X_train, y_train)

y_pred_lr = lr_classifier.predict(X_test)

lr_report = classification_report(y_test, y_pred_lr, output_dict=False)

print(lr_report)

import tensorflow as tf
from tensorflow import keras

model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

y_pred_prob = model.predict(X_test)
y_pred_dl = (y_pred_prob >= 0.5).astype(int)

dl_report = classification_report(y_test, y_pred_dl, output_dict=False)

print(dl_report)

import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report

model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=200, batch_size=32, verbose=0)

y_pred_prob = model.predict(X_test)
y_pred_mlp = (y_pred_prob >= 0.5).astype(int)

mlp_report = classification_report(y_test, y_pred_mlp, output_dict=False)

print(mlp_report)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

rf_classifier = RandomForestClassifier(n_estimators=50, random_state=42)  # Adjust n_estimators if needed
rf_classifier.fit(X_train, y_train)

y_pred_rf = rf_classifier.predict(X_test)

rf_report = classification_report(y_test, y_pred_rf, output_dict=False)

print(rf_report)

from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report

adaboost_classifier = AdaBoostClassifier(n_estimators=50, random_state=42)  # Adjust n_estimators if needed
adaboost_classifier.fit(X_train, y_train)

y_pred_adaboost = adaboost_classifier.predict(X_test)

adaboost_report = classification_report(y_test, y_pred_adaboost, output_dict=False)

print(adaboost_report)

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report

deep_forest_classifier = ExtraTreesClassifier(n_estimators=100, random_state=42)
deep_forest_classifier.fit(X_train, y_train)

y_pred_deep_forest = deep_forest_classifier.predict(X_test)

deep_forest_report = classification_report(y_test, y_pred_deep_forest, output_dict=False)

print(deep_forest_report)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import classification_report

dnn_model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

dnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

dnn_model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1, validation_split=0.2)

y_pred_dnn = (dnn_model.predict(X_test) > 0.5).astype(int).flatten()

dnn_report = classification_report(y_test, y_pred_dnn, output_dict=False)

print(dnn_report)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

advanced_ann = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.4),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

advanced_ann.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1),
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
]

advanced_ann.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=callbacks, verbose=1)

y_pred_advanced_ann = (advanced_ann.predict(X_test) >= 0.5).astype(int).flatten()

advanced_ann_report = classification_report(y_test, y_pred_advanced_ann, output_dict=False)

print(advanced_ann_report)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

softmax_regression = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=200, random_state=42)
softmax_regression.fit(X_train, y_train)

y_pred_softmax = softmax_regression.predict(X_test)

softmax_report = classification_report(y_test, y_pred_softmax, output_dict=False)

print(softmax_report)

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report

gbdt_classifier = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, max_depth=3, random_state=42)
gbdt_classifier.fit(X_train, y_train)

y_pred_gbdt = gbdt_classifier.predict(X_test)

gbdt_report = classification_report(y_test, y_pred_gbdt, output_dict=False)

print(gbdt_report)

reports = {
    "KNN": knn_report,
    "Naive Bayes": nb_report,
    "Support Vector Machine": svm_report,
    "Decision Tree": dt_report,
    "Logistic Regression": lr_report,
    "Deep Learning": dl_report,
    "Artificial neural network": advanced_ann_report,
    "Multi Layer Perceptron": mlp_report,
    "Random Forest": rf_report,
    "AdaBoost": adaboost_report,
    "Deep Forest": deep_forest_report,
    "Deep Neural Network": dnn_report,
    "Softmax Regression": softmax_report,
    "Gradient Boost Decision Tree": gbdt_report
}

metric = 'accuracy'

results = pd.DataFrame(columns=['Model', metric])

for model_name, report in reports.items():
    lines = report.split('\n')

    for line in lines:
        if 'accuracy' in line and line.strip().startswith('accuracy'):
            accuracy = float(line.split()[-2])
            results = pd.concat([results, pd.DataFrame({'Model': [model_name], metric: [accuracy]})], ignore_index=True)
            break
results = results.sort_values(by=[metric], ascending=False)

print(results)

if not results.empty:
    best_model = results.iloc[0]['Model']
    print(f"\nThe best model based on '{metric}' is: {best_model}")
else:
    print("\nNo accuracy results found in the reports.")