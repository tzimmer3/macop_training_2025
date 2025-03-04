# Functions for evaluating models
import pandas as pd
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, matthews_corrcoef

#==============================
# Confusion Matrix
#==============================
def confusion_matrix(target, prediction):
    """
    Creates a confusion matrix from a target column and a prediction column.
    """
    # Create data for a confusion matrix 
    confusion_matrix_data = pd.crosstab(target, prediction)
    # Confusion Matrix
    sns.heatmap(confusion_matrix_data, cmap='PuOr', annot=True,fmt=".1f",annot_kws={'size':16})


#==============================
# Categorical Accuracy Table
#==============================
def categorical_accuracy_table(target, prediction):
    """
    Creates a summary table with relevant accuracy metrics from a target column and prediction column.
    """

    precision, recall, f1_score, _ = precision_recall_fscore_support(
        target, prediction, average='weighted')
    accuracy = accuracy_score(prediction, target)
    mcc = matthews_corrcoef(prediction, target)

    measures = ['Accuracy','F1_Score','Precision', 'Recall', 'MCC']
    scores = [accuracy, f1_score, precision, recall, mcc]
    
    accuracytable = pd.DataFrame({'Measure': measures, 'Value': scores})
    
    return accuracytable