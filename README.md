# Iris Species Classification using KNN

This project is part of my AI & ML Internship Task 6 at Elevate Labs.

---

## Objective

- Classify Iris flower species using the **K-Nearest Neighbors (KNN)** algorithm.
- Normalize features to ensure fair distance calculations.
- Try different values of **K** and evaluate the model.
- Visualize decision boundaries to understand how KNN separates the classes.

---

## Dataset

- **Iris.csv**: Classic Iris flower dataset containing:
  - Sepal length & width
  - Petal length & width
  - Target species: Setosa, Versicolor, Virginica

---

## Steps Performed

1. **Data Loading & Cleaning**
   - Loaded the dataset using Pandas.
   - Dropped the `Id` column (not useful for prediction).

2. **Feature Engineering**
   - Normalized all features using `StandardScaler`.
   - Encoded target `Species` into integers using `LabelEncoder`.

3. **Train-Test Split**
   - Split data into 80% training and 20% testing using stratification to maintain class balance.

4. **Model Training & Evaluation**
   - Trained KNN models with `K=3`, `5`, and `7`.
   - Evaluated using:
     - Accuracy
     - Confusion Matrix
     - Precision, Recall, F1-score

5. **Visualization**
   - Plotted **decision boundaries** using the first two features to see how KNN separates different species.

---

## Results Summary

| K | Accuracy |
|---|----------|
| 3 | ~97%     |
| 5 | ~97%     |
| 7 | ~97%     |

KNN performed consistently well across different `K` values on this dataset.

---

## Tech Stack

- **Python**
- **Pandas, NumPy** for data handling
- **Matplotlib, Seaborn** for visualization
- **Scikit-learn** for preprocessing, KNN, and metrics

---

## How to Run

1. Install required packages:
pip install pandas numpy matplotlib seaborn scikit-learn

2. Run the script:
python task6.py

---

## Created by

**Yogesh Rajput**  

