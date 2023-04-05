import os
import argparse
from typing import Callable

import pandas as pd
import lightgbm as lgb
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

import mlflow

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

# メトリック記録のための callback 定義
def log_metrics() -> Callable[[lgb.callback.CallbackEnv], None]:
    def _callback(env: lgb.callback.CallbackEnv) -> None:
        mlflow.log_metric(env.evaluation_result_list[0][1], env.evaluation_result_list[0][2])
        print(f"iteration {env.iteration} {env.evaluation_result_list[0][1]}: {env.evaluation_result_list[0][2]}")
    _callback.order = 10
    return _callback

# pyfuncのモデルラッパー定義
class LGBWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        import lightgbm as lgb
        self.lgb_model = lgb.Booster(model_file=context.artifacts["lgb_model_path"])

    def predict(self, context, model_input):
        return self.lgb_model.predict(model_input)

# ローカル動作時とリモート動作時の差分を吸収する関数
def set_run(mode):
    if mode == 'remote':
        print("skip setting mlflow tracking uri")
    elif mode == 'local':
        print("setting mlflow tracking uri...")
        subscription_id = "SUBSCRIPTION_ID"
        resource_group = "RESOURCE_GROUP"
        workspace = "AML_WORKSPACE_NAME"

        ml_client = MLClient(
            DefaultAzureCredential(),
            subscription_id,
            resource_group,
            workspace,
        )

        azureml_mlflow_uri = ml_client.workspaces.get(
            ml_client.workspace_name
        ).mlflow_tracking_uri

        mlflow.set_tracking_uri(azureml_mlflow_uri)

        print("complete setting mlflow tracking uri")

        exp = mlflow.set_experiment("chapter5-lightgbm-job")
    
    run = mlflow.start_run()

    return run

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default='local')
    args = parser.parse_args()
    # データをpandas データフレームとして読み込み
    print("loading data...")
    credit_df = pd.read_excel(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls",
        header=1,
        index_col=0
    )

    # 分割
    _df, test_df = train_test_split(
        credit_df,
        test_size=0.2,
    )

    train_df, valid_df = train_test_split(
        _df,
        test_size=0.2,
    )

    # 加工
    y_train = train_df.pop("default payment next month")
    X_train = train_df.values
    train_dataset = lgb.Dataset(X_train, label=y_train)

    y_valid = valid_df.pop("default payment next month")
    X_valid = valid_df.values
    valid_dataset = lgb.Dataset(X_valid, label=y_valid)

    y_test = test_df.pop("default payment next month")
    X_test = test_df.values

    print("complete data preparation")

    run = set_run(args.mode)

    params = {
        "objective": "binary",
        "boosting_type": "gbdt",
        "num_leaves": 20,
        "max_depth": 10,
        "learning_rate": 0.1,
        "device_type": "cpu",
        "seed": 42,
        "deterministic": True,
    }
    mlflow.log_params(params)

    print("training...")

    clf = lgb.train(
        params,
        train_set=train_dataset,
        valid_sets=[valid_dataset],
        valid_names=['valid'],
        callbacks=[
            log_metrics(),
            lgb.early_stopping(stopping_rounds=10, verbose=True)
        ]
    )

    print("complete training...")

    y_prob = clf.predict(X_test)
    y_pred = [1 if y_prob >= 0.5 else 0 for y_prob in y_prob]

    result = classification_report(y_test, y_pred, output_dict=True)
    print(result)
    mlflow.log_metrics(result["0"])

    model_path = "model.txt"
    clf.save_model(model_path)

    artifacts = {"lgb_model_path": model_path}

    signature = mlflow.models.signature.infer_signature(X_test, y_prob)

    mlflow_model_dir = 'lgb_model'
    mlflow.pyfunc.log_model(
        artifact_path=mlflow_model_dir,
        python_model=LGBWrapper(),
        conda_env='environment.yaml',
        artifacts=artifacts,
        signature=signature,
    )

    mlflow.end_run()

    mlflow.register_model(
        model_uri=f"runs:/{run.info.run_id}/{mlflow_model_dir}/",
        name='chapter5-pyfunc-model'
    )

if __name__ == "__main__":
    main()