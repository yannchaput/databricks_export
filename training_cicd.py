import argparse
import sys
import mlflow
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from mlflow.tracking import MlflowClient

MODEL_NAME="ElasticnetWineModel"

def parse_args():
  parser = argparse.ArgumentParser(description="machine learning industrialized training example")
  parser.add_argument("alpha",type=float,help="Elasticnet alpha hyperparamter",default=0.1)
  parser.add_argument("l1_ratio",type=float,help="Elasticnet l1 regularization ratio",default=0.5)
  args = parser.parse_args()
  return args

def create_experiment() -> str:
  exp_path="/Repos/yann.chaput@axa.com/databricks-projects/notebooks/mlflow/train_cicd_exp"
  exp=mlflow.get_experiment_by_name(exp_path)
  exp_id=None
  if exp is None:
    exp_id=mlflow.create_experiment(exp_path)
  else:
    exp_id=exp.experiment_id
  return exp_id

def clean_models(model: str):
  mlflow_client = MlflowClient()
  # Delete a registered model along with all its versions
  mlflow_client.delete_registered_model(name=model)

  
# train and register model
args=parse_args()
alpha=args.alpha
l1_ratio=args.l1_ratio
# create new experiment if not existing
exp_id=create_experiment()
print("Using experiment {}".format(exp_id))
clean_models(MODEL_NAME)
submitted_run = mlflow.projects.run(uri="https://github.com/mlflow/mlflow#examples/sklearn_elasticnet_wine", experiment_id=exp_id, parameters={"alpha":alpha, "l1_ratio":l1_ratio})
submitted_run.wait()
print(f"Submitted run with id: {submitted_run.run_id} and status: {submitted_run.get_status()}")

# Evaluate model (should be on unseen test data)
csv_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
train_ds = pd.read_csv(csv_url,delimiter=";",header=0)
y_actual = train_ds.pop("quality")
logged_model = "models:/{model_name}/none".format(model_name=MODEL_NAME)
# Load model as a PyFuncModel.
loaded_model = mlflow.sklearn.load_model(logged_model)
# Predict on a Pandas DataFrame.
y_predicted = loaded_model.predict(train_ds)
r2 = r2_score(y_actual,y_predicted)
sys.exit(str(r2))
