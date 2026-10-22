
# KNN Handwritten Digit Classifier

A Python machine learning project that recognises handwritten digits using a K-Nearest Neighbours (KNN) classifier implemented from scratch.

## Project Overview

This project applies machine learning concepts learned at university to a practical digit recognition task.

Dataset: Scikit-learn Digits (1,797 images, 8×8 pixels)

## Features

- Implemented L1 (Manhattan) and L2 (Euclidean) distance calculations
- Built KNN classification using nearest-neighbour ranking and majority voting
- Compared K values of 1, 3 and 5
- Visualised model accuracy with Matplotlib
- Investigated incorrect predictions by examining neighbouring labels and distances

## Results

| K Value | Accuracy (50 test images) |
|---------|---------------------------|
| K = 1   | 100%                      |
| K = 3   | 100%                      |
| K = 5   | 98%                       |

For K=5, the model incorrectly classified a handwritten 6 as 8.

The five nearest labels were [6, 8, 6, 8, 8]. This showed how increasing K can change the majority vote and the final prediction.

These results are from the first 50 images of a held-out test set, not the entire dataset. Future work will use a separate validation set for parameter selection and evaluate performance on the full test set.

## Technologies

- Python
- NumPy
- Matplotlib
- Scikit-learn (dataset and train/test split)

## How to Run

1. Install dependencies:

   `pip install -r requirements.txt`

2. Run the program:

   `python main.py`

## Key Learnings

- Understanding the relationship between mathematical theory and Python implementation
- How K and distance metrics affect nearest-neighbour classification
- Evaluating machine learning predictions
- Debugging, visualisation and error analysis

## Future Improvements

- Evaluate performance on the complete test set
- Compare L1 and L2 accuracy
- Add cross-validation for hyperparameter selection
- Build an interactive prediction interface
