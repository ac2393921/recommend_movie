import mlflow

with mlflow.start_run():
    for epoch in range(0, 3):
        mlflow.log_metric(key="train acc", value=2 * epoch, step=epoch)
