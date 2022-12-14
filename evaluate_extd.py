import json
import logging
import pathlib
import pickle
import tarfile
import numpy as np
import pandas as pd
import xgboost

from sklearn.metrics import(
    accuracy_score,
#     precision_score,
#     recall_score,
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
    model_path = "/opt/ml/processing/model/model.tar.gz"
    with tarfile.open(model_path) as tar:
        tar.extractall(path="..")
        
    logger.debug("Loading xgboost model")
    model = pickle.load(open("xgboost-model", "rb"))
    
    logger.debug("Loading test input data")
    test_path = "/opt/ml/processing/test/test_feats.csv"
    df = pd.read_csv(test_path)
    
    logger.debug("Reading test data")
    y_test = df.iloc[:, 0].to_numpy()
    df.drop(df.columns[0], axis=1, inplace=True)
    X_test = xgboost.DMatrix(df.values)
    
    logger.info("Performing predictions against test data")
    prediction_probabilities = model.predict(X_test)
    predictions = np.round(prediction_probabilities)
    
#     precision = precision_score(y_test, predictions)
#     recall = recall_score(y_test, predictions)
    accuracy = accuracy_score(y_test, predictions)
    conf_matrix = confusion_matrix(y_test, predictions)
    fpr, tpr, _ = roc_curve(y_test, prediction_probabilities)
    
    auc_score = auc(fpr, tpr) 
    pr_curve = precision_recall_curve(y_test, predictions)
    precision = precision_score(y_test, predictions)
    avg_precision = average_precision_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, predictions)
#     log_loss_score = log_loss(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    roc = roc_curve(y_test, predictions)
    informedness = balanced_accuracy_score(y_test, predictions, adjusted=True)
    cohen_kappa = cohen_kappa_score(y_test, predictions)
    matthews_coef = matthews_corrcoef(y_test, predictions)
    fbeta = fbeta_score(y_test, predictions, beta=0.5)
        
    logger.debug("Accuracy: {}".format(accuracy))
#     logger.debug("Precision: {}".format(precision))
#     logger.debug("Recall: {}".format(recall))
    logger.debug("Confusion matrix: {}".format(conf_matrix))
    
    logger.debug("AUC: {}".format(auc_score))
    logger.debug("Precision Recall Curve: {}".format(pr_curve))
    logger.debug("Precision: {}".format(precision))
    logger.debug("Average Percision: {}".format(avg_precision))
    logger.debug("ROC AUC: {}".format(roc_auc))
#     logger.debug("Log loss: {}".format(log_loss_score))
    logger.debug("F1: {}".format(f1))
    logger.debug("Recall: {}".format(recall))
    logger.debug("ROC: {}".format(roc))
    logger.debug("informedness: {}".format(informedness))
    logger.debug("Cohen Kappa: {}".format(cohen_kappa))
    logger.debug("Mathews Correlation Coefficient: {}".format(matthews_coef))
    logger.debug("Fbeta: {}".format(fbeta))
    
    report_dict = {
        "binary_classification_metrics": {
            "auc": {"value":auc_score, "standard_deviation":"NaN"},
            "precision": {"value":precision, "standard_deviation":"NaN"},
            "avg_percision": {"value":avg_precision, "standard_deviation":"NaN"},
            "roc_auc": {"value":roc_auc, "standard_deviation":"NaN"},
#             "log_loss": {"value":log_loss_score, "standard_deviation":"NaN"},
            "f1": {"value":f1, "standard_deviation":"NaN"},
            "recall": {"value":recall, "standard_deviation":"NaN"},
#             "roc": {"value":roc, "standard_deviation":"NaN"},
            "informedness": {"value":informedness, "standard_deviation":"NaN"},
            "cohen_kappa": {"value":cohen_kappa, "standard_deviation":"NaN"},
            "mathews_coef": {"value":matthews_coef, "standard_deviation":"NaN"},
            "fbeta": {"value":fbeta, "standard_deviation":"NaN"},
            "accuracy": {"value":accuracy, "standard_deviation":"NaN"},
#             "pr_curve": {"0": {"0": int(pr_curve[0][0]), "1": int(pr_curve[0][1])},
#                          "1": {"0": int(pr_curve[1][0]), "1": int(pr_curve[1][1])},
            "confusion_matrix": {"0": {"0": int(conf_matrix[0][0]), "1": int(conf_matrix[0][1])},
                                 "1": {"0": int(conf_matrix[1][0]), "1": int(conf_matrix[1][1])}
                                },
            "receiver_operating_charastic_curve": {
                "false_positive_rates": list(fpr),
                "true_positive_rates": list(tpr)
            }
        }
    }
    
    print(report_dict)
    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    evaluation_path = f'{output_dir}/evaluation.json'
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))        
