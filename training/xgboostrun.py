import modelcommon

try:
    from xgboost import XGBRegressor
except ModuleNotFoundError as error:
    raise SystemExit(
        "xgboost is not installed. Run `python -m pip install xgboost` and rerun xgboostrun.py."
    ) from error


def buildruns():
    return [
        {
            "n_estimators": 350,
            "max_depth": 3,
            "learning_rate": 0.06,
            "min_child_weight": 2,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "reg_lambda": 1.0,
        },
        {
            "n_estimators": 500,
            "max_depth": 4,
            "learning_rate": 0.04,
            "min_child_weight": 3,
            "subsample": 0.85,
            "colsample_bytree": 0.9,
            "reg_lambda": 1.4,
        },
        {
            "n_estimators": 650,
            "max_depth": 5,
            "learning_rate": 0.035,
            "min_child_weight": 4,
            "subsample": 0.8,
            "colsample_bytree": 0.85,
            "reg_lambda": 1.8,
        },
        {
            "n_estimators": 280,
            "max_depth": 2,
            "learning_rate": 0.08,
            "min_child_weight": 1,
            "subsample": 0.95,
            "colsample_bytree": 0.95,
            "reg_lambda": 0.8,
        },
        {
            "n_estimators": 450,
            "max_depth": 4,
            "learning_rate": 0.05,
            "min_child_weight": 2,
            "subsample": 0.88,
            "colsample_bytree": 0.82,
            "reg_lambda": 1.2,
        },
    ]


def main():
    split_data = modelcommon.loadandsplit()
    best_payload = None

    for run_number, settings in enumerate(buildruns(), start=1):
        model = XGBRegressor(
            objective="reg:squarederror",
            tree_method="hist",
            random_state=run_number * 13,
            early_stopping_rounds=40,
            **settings,
        )

        model.fit(
            split_data["x_train"],
            split_data["y_train"],
            eval_set=[(split_data["x_valid"], split_data["y_valid"])],
            verbose=False,
        )

        predictions = model.predict(split_data["x_valid"])
        metrics = modelcommon.metricsfrom(split_data["y_valid"], predictions)

        payload = {
            "model_name": "XGBRegressor",
            "run_number": run_number,
            "settings": settings,
            "metrics": metrics,
            "best_iteration": int(model.best_iteration) if model.best_iteration is not None else None,
        }

        if best_payload is None or payload["metrics"]["rmse"] < best_payload["metrics"]["rmse"]:
            best_payload = payload

    modelcommon.writeresult("xgboostrun_results.json", best_payload)
    modelcommon.printresult("XGBRegressor", best_payload)
    print(f"best settings: {best_payload['settings']}")


if __name__ == "__main__":
    main()
