# Iris Flower Classification

A beginner-friendly machine-learning project that classifies Iris flowers as Setosa, Versicolor, or Virginica using measurements of their sepals and petals.

## Tools used
- Python
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

## Dataset
The project uses the Iris dataset bundled with scikit-learn, so a separate dataset file is not needed.

## How to run
1. Install Python 3.9 or newer.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the script from this repository folder:
   ```bash
   python src/iris_classification.py
   ```

## What the script does
1. Loads and previews the Iris dataset.
2. Splits the data into training and testing sets.
3. Trains a Logistic Regression classifier.
4. Prints accuracy and a classification report.
5. Saves a petal-measurement scatter plot, confusion matrix, and text report to `outputs/`.

## Output files
- `outputs/iris_scatter_plot.png` — scatter plot of petal length and width by species.
- `outputs/confusion_matrix.png` — actual vs predicted classes.
- `outputs/classification_report.txt` — accuracy, precision, recall, and F1-score.

## Notes
The train/test split uses a fixed random seed for repeatable results. This is an educational example; performance metrics can vary if you change the split or model.
