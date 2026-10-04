# Iris Flower Classifier

A machine learning project that uses a Decision Tree to predict
Iris flower species from four flower measurements.

## Tools
Python, scikit-learn, matplotlib, pandas, numpy and Jupyter Notebook.

## Dataset and Results
- Iris dataset: 150 flowers across three species.
- Training set: 120 flowers.
- Test set: 30 flowers.
- Test accuracy: 93.3% (28 out of 30 correct).

## Project Files
- notebooks/iris_model.ipynb: interactive notebook.
- src/train.py: training and evaluation script.
- outputs/confusion_matrix.png: prediction results chart.
- requirements.txt: required Python packages.
- .gitignore: files and folders excluded from Git.

## Run on Windows
From the project folder:

    python -m venv venv
    venv\Scripts\activate
    python -m pip install -r requirements.txt
    python src\train.py

The script prints test accuracy and saves the confusion matrix chart.