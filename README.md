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
