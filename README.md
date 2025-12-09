# Churn Prediction using ANN

This project builds, trains, and evaluates an Artificial Neural Network (ANN) to predict customer churn using the classic `Churn_Modelling.csv` dataset. The workflow is implemented in the Jupyter notebook **Prediction.ipynb**.

---

## Project Structure

- `Prediction.ipynb` – main notebook containing the complete data processing, model training, and evaluation pipeline.
- `Churn_Modelling.csv` – input dataset (10,000 customers with churn labels).
- `README.md` – project documentation (this file).

You may optionally add:
- `models/` – to save trained models.
- `requirements.txt` – to pin dependencies.

---

## Dataset

The notebook loads the dataset:

```python
df = pd.read_csv('Churn_Modelling.csv')
```

Initial columns include:

- `RowNumber`
- `CustomerId`
- `Surname`
- `CreditScore`
- `Geography` (France, Germany, Spain)
- `Gender` (Male, Female)
- `Age`
- `Tenure`
- `Balance`
- `NumOfProducts`
- `HasCrCard`
- `IsActiveMember`
- `EstimatedSalary`
- `Exited` (target: 1 = churned, 0 = not churned)

Basic checks:

- `df.info()` – confirms 10,000 rows and 14 columns, with no missing values.
- `df.duplicated()` – verifies there are no duplicate rows.
- `df['Exited'].value_counts()` – shows class imbalance:
  - `0` (not churned): 7963
  - `1` (churned): 2037
- `df['Geography'].value_counts()` – France, Germany, Spain.
- `df['Gender'].value_counts()` – Male, Female.

---

## Exploratory & Data Preprocessing

### Column Dropping

The following identifier columns are removed as they are not predictive:

```python
df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)
```

### Categorical Encoding

`Geography` and `Gender` are converted to dummy variables:

```python
df['Geography'] = dff['Geography']  # Restore original Geography before encoding

df = pd.get_dummies(df, columns=['Geography', 'Gender'], drop_first=True)
```

Final feature columns include:

- `CreditScore`
- `Age`
- `Tenure`
- `Balance`
- `NumOfProducts`
- `HasCrCard`
- `IsActiveMember`
- `EstimatedSalary`
- `Exited` (target)
- `Geography_Germany` (bool)
- `Geography_Spain` (bool)
- `Gender_Male` (bool)

Total: **12 columns** (11 features + 1 target).

---

## Feature/Target Split and Train/Test Split

```python
X = df.drop('Exited', axis=1)
y = df['Exited']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

- Training set: 8,000 samples
- Test set: 2,000 samples

---

## Feature Scaling

All features are standardized using `StandardScaler`:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_trained_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

---

## Model Architecture (ANN)

The ANN is built using TensorFlow / Keras:

```python
import tensorflow
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(6, activation='sigmoid', input_dim=X_trained_scaled.shape[1]))
model.add(Dense(3, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))
```

Model summary:

- **Input layer**: 11 features
- **Hidden Layer 1**: 6 neurons, `sigmoid`
- **Hidden Layer 2**: 3 neurons, `sigmoid`
- **Output Layer**: 1 neuron, `sigmoid` (binary classification)
- **Total parameters**: 97

Compilation:

```python
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)
```

---

## Training

The model is trained for 100 epochs with validation split:

```python
history = model.fit(
    X_trained_scaled,
    y_train,
    epochs=100,
    validation_split=0.2
)
```

Plots generated in the notebook:

- Training vs Validation **Loss**
- Training vs Validation **Accuracy**

These show how the model converges and whether overfitting occurs.

---

## Evaluation

### Predictions

```python
y_log = model.predict(X_test_scaled)
y_pred = np.where(y_log > 0.5, 1, 0)
```

### Metrics

```python
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
```

Reported results:

- **Accuracy**: `0.8605` (86.05%)

Classification report:

- Class `0` (not churned):
  - Precision: 0.88
  - Recall: 0.96
  - F1-score: 0.92
  - Support: 1607
- Class `1` (churned):
  - Precision: 0.73
  - Recall: 0.46
  - F1-score: 0.56
  - Support: 393

Overall:

- Accuracy: 0.86
- Macro avg:
  - Precision: 0.81
  - Recall: 0.71
  - F1-score: 0.74
- Weighted avg:
  - Precision: 0.85
  - Recall: 0.86
  - F1-score: 0.85

Confusion matrix:

```text
[[1541   66]
 [ 213  180]]
```

Interpretation:

- True negatives: 1,541
- False positives: 66
- False negatives: 213
- True positives: 180

The model performs very well on non-churn customers but has lower recall on churned customers (misses some churns).

---

## Training Curves

The notebook plots:

```python
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()
```

These curves help visualize:

- Convergence behavior
- Gap between training and validation performance
- Potential overfitting or underfitting

---

## Dependencies

Main libraries used:

- Python 3.13
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- tensorflow / keras

Example `requirements.txt` (adjust versions as needed):

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
tensorflow
jupyter
```

---

## How to Run

1. **Clone the repository**:

   ```bash
   git clone https://github.com/CresthaRaman/Churn-Prediction-using-ANN.git
   cd Churn-Prediction-using-ANN
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

   or manually install:

   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn tensorflow jupyter
   ```

3. **Place the dataset**:

   Ensure `Churn_Modelling.csv` is in the project root (same folder as `Prediction.ipynb`), or update the path in the notebook.

4. **Run the notebook**:

   ```bash
   jupyter notebook Prediction.ipynb
   ```

   Execute all cells in order to:
   - Load and preprocess data
   - Train the ANN
   - Evaluate and visualize performance

---

## Possible Improvements

- Address class imbalance (e.g., class weights, SMOTE, or resampling).
- Try different architectures:
  - More layers / neurons
  - `relu` activations
  - Dropout or batch normalization.
- Hyperparameter tuning:
  - Learning rate, batch size, epochs.
- Additional evaluation metrics:
  - ROC-AUC, Precision-Recall curves.

---

## License

This project is available under an open-source license (add the specific license here if you choose, e.g., MIT License).

---

## Author

- GitHub: [CresthaRaman](https://github.com/CresthaRaman)
