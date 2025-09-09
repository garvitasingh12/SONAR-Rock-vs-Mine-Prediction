# Rock vs Mine Prediction using Sonar Data

A machine learning project that predicts whether an object detected by sonar signals is a **rock** or a **mine**. This project uses the classic **Sonar dataset** and applies **Logistic Regression** for binary classification.

---

## 📖 Table of Contents
- [About](#about)
- [Background](#background)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Installation](#installation)
- [Usage](#usage)
- [Code Walkthrough](#code-walkthrough)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Results](#results)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)
- [Author](#author)

---

## About
Sonar signals are widely used in detecting underwater objects. Correctly identifying whether the detected object is a rock formation or a mine is crucial for safety and defense applications.  

This project demonstrates:
- Loading and preprocessing a real-world dataset
- Training a Logistic Regression classification model
- Evaluating model accuracy on training and test data
- Making predictions for new sonar signal data

---

## Background
The **Sonar dataset** is a well-known benchmark dataset from the UCI Machine Learning Repository. Each instance in the dataset is a set of 60 numerical features representing sonar signal energy levels at different frequencies. The task is to classify the signals as either:
- **R (Rock)** → sonar bounced back from a rock
- **M (Mine)** → sonar bounced back from a metal cylinder (mine)

This classification problem is particularly relevant for **marine navigation, defense systems, and safety applications**.

---

## Dataset
- **Name**: Sonar, Mines vs Rocks Dataset  
- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/connectionist+bench+(sonar,+mines+vs.+rocks))  
- **Description**:  
  - 208 samples  
  - 60 numeric features (sonar signal strength at different frequencies)  
  - 1 label column:  
    - `R` → Rock  
    - `M` → Mine  

---

## Project Workflow
The workflow of this project is as follows:

1. **Data Loading**
   - Load the dataset into a Pandas DataFrame.
   - Inspect dataset shape and columns.

2. **Exploratory Data Analysis (EDA)**
   - View summary statistics using `.describe()`.
   - Analyze class distribution using `.value_counts()`.
   - Compute class-wise mean feature values.

3. **Preprocessing**
   - Separate the dataset into features (`X`) and labels (`Y`).
   - Split the data into training and test sets (90% training, 10% testing) using stratified sampling.

4. **Model Training**
   - Initialize a **Logistic Regression** model.
   - Fit the model on the training set.

5. **Model Evaluation**
   - Predict labels for training and test sets.
   - Compute accuracy using `sklearn.metrics.accuracy_score`.

6. **Prediction**
   - Provide custom input data as a tuple of 60 feature values.
   - Convert input to a NumPy array, reshape, and predict with the trained model.

---

## Installation
Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/sonar-rock-vs-mine.git
cd sonar-rock-vs-mine
pip install -r requirements.txt
```

---

## Usage
### Running in Google Colab
1. Open the notebook in Google Colab.  
2. Upload the dataset file (`Copy of sonar data.csv`).  
3. Run the notebook cells in order.  

### Running Locally
1. Ensure Python 3.8+ is installed.  
2. Place `sonar_data.csv` inside a `data/` folder.  
3. Run the notebook using Jupyter:
   ```bash
   jupyter notebook notebooks/Rock_vs_Mine_prediction.ipynb
   ```

---

## Code Walkthrough

### Load Dataset
```python
import pandas as pd
sonar_data = pd.read_csv("data/sonar_data.csv", header=None)
```

### Split Features and Labels
```python
X = sonar_data.drop(columns=60, axis=1)
Y = sonar_data[60]
```

### Train-Test Split
```python
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.1, stratify=Y, random_state=1
)
```

### Train Model
```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, Y_train)
```

### Evaluate Accuracy
```python
from sklearn.metrics import accuracy_score
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

print("Training Accuracy:", accuracy_score(train_pred, Y_train))
print("Testing Accuracy:", accuracy_score(test_pred, Y_test))
```

### Predict on New Data
```python
import numpy as np
input_data = (0.02,0.037,...,0.011)  # example 60 values
input_array = np.asarray(input_data).reshape(1, -1)
prediction = model.predict(input_array)
print("Predicted Class:", prediction[0])  # 'R' or 'M'
```

---

## Project Structure
```
├── data/
│   └── sonar_data.csv
├── notebooks/
│   └── Rock_vs_Mine_prediction.ipynb
├── requirements.txt
└── README.md
```

---

## Requirements
The project dependencies are listed below:

```
numpy
pandas
scikit-learn
```

If you want exact versions for reproducibility, you can freeze your environment with:
```bash
pip freeze > requirements.txt
```

---

## Results
- **Training accuracy**: ~83%  
- **Testing accuracy**: ~76%  
- The model successfully predicts whether sonar signals correspond to a rock or a mine.

---

## Future Improvements
- Experiment with other classifiers (SVM, Random Forest, Neural Networks).  
- Apply feature scaling (StandardScaler, MinMaxScaler).  
- Perform hyperparameter tuning (GridSearchCV, RandomizedSearchCV).  
- Implement cross-validation for robust performance estimation.  
- Deploy the model as a **web app** using Streamlit or Flask.  

---

## Contributing
Contributions are welcome!  
If you’d like to improve this project:
1. Fork the repo  
2. Create a new branch (`git checkout -b feature-name`)  
3. Commit changes (`git commit -m 'Add feature'`)  
4. Push branch (`git push origin feature-name`)  
5. Create a Pull Request  

---

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Author
- **Garvita Singh**
