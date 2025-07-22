# K-Nearest Neighbors (KNN) Classifier – Python (From Scratch)

A simple and intuitive implementation of the K-Nearest Neighbors (KNN) classification algorithm using only core Python. This project demonstrates the fundamentals of machine learning by building a model without relying on external libraries like NumPy or scikit-learn.

##  Features

- Pure Python (no external libraries)
- Uses **Euclidean distance** as the similarity metric
- **Majority voting** for classification
- Supports **multi-class classification**
- Adjustable `k` value for nearest neighbors
- Clean and modular function design

## Algorithm Overview

1. **Distance Calculation**: Computes Euclidean distance between test and training samples.
2. **Neighbor Selection**: Sorts all training samples by distance and picks top-k.
3. **Prediction**: Assigns the class with the highest frequency among neighbors.

## 🔧 Functions

- `euclidean_distance(row1, row2)` – Calculates distance between two data points.
- `get_neighbors(train, test_row, k)` – Returns k nearest neighbors.
- `predict_classification(train, test_row, k)` – Predicts the class label.
- `k_nearest_neighbors(train, test, k)` – Main function to classify test data.

## Example Usage

```python
train_data = [
    [2.0, 4.0, 'A'],
    [1.0, 3.0, 'A'],
    [5.0, 7.0, 'B'],
    [6.0, 8.0, 'B']
]

test_data = [
    [3.0, 5.0],
    [5.5, 7.5]
]

predictions = k_nearest_neighbors(train_data, test_data, k=3)
print(predictions)
