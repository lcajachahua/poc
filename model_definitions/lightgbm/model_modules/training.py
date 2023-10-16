from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
from nyoka import lgb_to_pmml
from teradataml import DataFrame
from aoa import (
    record_training_stats,
    save_plot,
    aoa_create_context,
    ModelContext
)

import joblib


def train(context: ModelContext, **kwargs):
    aoa_create_context()

    feature_names = context.dataset_info.feature_names
    target_name = context.dataset_info.target_names[0]

    # read training dataset from Teradata and convert to pandas
    train_df = DataFrame.from_query(context.dataset_info.sql)
    train_pdf = train_df.to_pandas(all_rows=True)

    # split data into X and y
    X_train = train_pdf[feature_names]
    y_train = train_pdf[target_name]

    print("Starting training...")

    # fit model to training data
    model = Pipeline([('lgbmc', LGBMClassifier(objective = "binary", learning_rate=context.hyperparams["learning_rate"], max_depth=context.hyperparams["max_depth"], num_leaves=context.hyperparams["num_leaves"]))])

    model.fit(X_train, y_train)

    print("Finished training")

    # export model artefacts
    joblib.dump(model, f"{context.artifact_output_path}/model.joblib")

    # we can also save as pmml so it can be used for In-Vantage scoring etc.
    lgb_to_pmml(pipeline=model, col_names=feature_names, target_name=target_name,
                    pmml_f_name=f"{context.artifact_output_path}/model.pmml")

    print("Saved trained model")

    record_training_stats(train_df,
                          features=feature_names,
                          targets=[target_name],
                          categorical=[target_name],
                          context=context)
