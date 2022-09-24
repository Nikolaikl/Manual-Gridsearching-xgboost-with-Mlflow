from argparse import ArgumentParser

import pandas as pd 

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
import xgboost as xgb
import matplotlib as mpl


import mlflow
import mlflow.xgboost


import logging

from urllib.parse import urlparse

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

mpl.use("Agg")


def parse_args():
    
    parser = ArgumentParser(description="XGBoost example")
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.3,
        help="learning rate to update step size at each boosting step (default: 0.3)",
    )
    parser.add_argument(
        "--colsample-bytree",
        type=float,
        default=1.0,
        help="subsample ratio of columns when constructing each tree (default: 1.0)",
    )
    parser.add_argument(
        "--subsample",
        type=float,
        default=1.0,
        help="subsample ratio of the training instances (default: 1.0)",
    )
    parser.add_argument(
        "--max_depth",
        type=int,
        default=10,
        help="Max Depth of the trees",
    )
    parser.add_argument(
        "--n_estimators",
        type=int,
        default=100,
        help="number of weak learners",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0,
        help="Regularisation for tree pruning",
    )
    parser.add_argument(
        "--reg_alpha",
        type=float,
        default=0,
        help="L1 regularization",
    )
    parser.add_argument(
        "--reg_lambda",
        type=float,
        default=1,
        help="L2 regularization",
    )
    
    return parser.parse_args()


def main():
    # parse command-line arguments
    args = parse_args()

    # prepare train and test data
    data = pd.read_csv("data/augmented_data.csv")
    
    # X features are 
    # 'grade', 'annual_inc', 'short_emp', 'emp_length_num', 'home_ownership',
    # 'dti', 'purpose', 'term', 'last_delinq_none', 'last_major_derog_none',
    # 'revol_util', 'total_rec_late_fee', 'od_ratio', 
    # y = 'bad_loan'
    
    X = data.drop(columns=['bad_loan'])

    y = data.bad_loan
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # enable auto logging
    mlflow.xgboost.autolog()

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    with mlflow.start_run():

        # train model
        params = {
            "objective": "binary:logistic",
            "learning_rate": args.learning_rate,
            "colsample_bytree": args.colsample_bytree,
            "subsample": args.subsample,
            "max_depth": args.max_depth,
            "n_estimators": args.n_estimators,
            "gamma": args.gamma,
            "reg_alpha": args.reg_alpha,
            "reg_lambda": args.reg_lambda,
            "seed": 42
        }
        
        model = xgb.XGBClassifier(**params) 
        
        model.fit(X_train, y_train)

        # evaluate model
        y_proba = model.predict_proba(X_test)
        y_pred = y_proba.argmax(axis=1)
        loss = log_loss(y_test, y_proba)
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred)

        # log metrics
        mlflow.log_metrics({"log_loss": loss, "accuracy": acc, "auc": auc})
        
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        if tracking_url_type_store != "file":

            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(model, "model", registered_model_name="xgb_credit")
        
        else:
            mlflow.sklearn.log_model(model, "model")


if __name__ == "__main__":
    main()