import json
import os
import logging
import pathlib
import pickle
import tarfile
import numpy as np
import pandas as pd

from sklearn.metrics import(
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_curve,
#     mine
    auc, 
    precision_recall_curve, 
    precision_score,
    average_precision_score,
    roc_auc_score,
    log_loss,
    f1_score,
    recall_score,
    roc_curve,
    make_scorer,
    balanced_accuracy_score,
    cohen_kappa_score,
    matthews_corrcoef,
    fbeta_score)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


if __name__ == "__main__":
    y_pred_path = "/opt/ml/processing/input/predictions/test_x.csv.out"
    y_true_path = "/opt/ml/processing/input/true_labels/test_y.csv"
    predictions = pd.read_csv(y_pred_path, header=None)
    y_test = pd.read_csv(y_true_path, header=None)
    
    accuracy = accuracy_score(y_test, predictions)
    conf_matrix = confusion_matrix(y_test, predictions)
    pr_curve = precision_recall_curve(y_test, predictions)
    precision = precision_score(y_test, predictions)
    avg_precision = average_precision_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    roc = roc_curve(y_test, predictions)
    fpr, tpr, thresholds = roc_curve(y_test, predictions)
    informedness = balanced_accuracy_score(y_test, predictions, adjusted=True)
    cohen_kappa = cohen_kappa_score(y_test, predictions)
    matthews_coef = matthews_corrcoef(y_test, predictions)
    fbeta = fbeta_score(y_test, predictions, beta=0.5)
        
    print("Accuracy: {}".format(accuracy))
    print("Confusion matrix: {}".format(conf_matrix))
    print("Precision Recall Curve: {}".format(pr_curve))
    print("Precision: {}".format(precision))
    print("Average Percision: {}".format(avg_precision))
    print("ROC AUC: {}".format(roc_auc))
    print("F1: {}".format(f1))
    print("Recall: {}".format(recall))
    print("ROC: {}".format(roc))
    print("informedness: {}".format(informedness))
    print("Cohen Kappa: {}".format(cohen_kappa))
    print("Mathews Correlation Coefficient: {}".format(matthews_coef))
    print("Fbeta: {}".format(fbeta))
    
    try:
        auc_score = auc(fpr, tpr)
        print("AUC: {}".format(auc_score))
    except:
        print("AUC doesn't work")
    try:
        log_loss_score = log_loss(y_test, predictions)
        print("Log loss: {}".format(log_loss_score))
    except:
        print("log_loss doesn't work")
    
    
    report_dict = {
        "binary_classification_metrics": {
            # "auc": {"value":auc_score, "standard_deviation":"NaN"},
            "precision": {"value":precision, "standard_deviation":"NaN"},
            "avg_percision": {"value":avg_precision, "standard_deviation":"NaN"},
            "roc_auc": {"value":roc_auc, "standard_deviation":"NaN"},
            # "log_loss": {"value":log_loss_score, "standard_deviation":"NaN"},
            "f1": {"value":f1, "standard_deviation":"NaN"},
            "recall": {"value":recall, "standard_deviation":"NaN"},
            "informedness": {"value":informedness, "standard_deviation":"NaN"},
            "cohen_kappa": {"value":cohen_kappa, "standard_deviation":"NaN"},
            "mathews_coef": {"value":matthews_coef, "standard_deviation":"NaN"},
            "fbeta": {"value":fbeta, "standard_deviation":"NaN"},
            "accuracy": {"value":accuracy, "standard_deviation":"NaN"},
            # "roc": {
            #     "fpr": roc[0].tolist(),
            #     "tpr": roc[1].tolist(),
            #     "thresholds": roc[2].tolist()}
            # "pr_curve": {"precision": pr_curve[0].tolist(),
            #              "recall": pr_curve[1].tolist(),
            #              "thresholds": pr_curve[2].tolist()},
            "confusion_matrix": {"0": {"0": int(conf_matrix[0][0]), "1": int(conf_matrix[0][1])},
                                 "1": {"0": int(conf_matrix[1][0]), "1": int(conf_matrix[1][1])}
                                },
            # "receiver_operating_charastic_curve": {
            #     "false_positive_rates": list(fpr),
            #     "true_positive_rates": list(tpr)
            # }
        }
    }
    
    print(report_dict)
    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    evaluation_path = os.path.join(output_dir, 'evaluation.json')
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))
