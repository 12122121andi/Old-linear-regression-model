from sklearn.neural_network import MLPRegressor

import modelcommon


def buildruns():
    return [
        {"hidden_layer_sizes": (48,), "alpha": 0.0005, "learning_rate_init": 0.003},
        {"hidden_layer_sizes": (64, 32), "alpha": 0.0005, "learning_rate_init": 0.0015},
        {"hidden_layer_sizes": (96, 48), "alpha": 0.001, "learning_rate_init": 0.001},
        {"hidden_layer_sizes": (128, 64), "alpha": 0.002, "learning_rate_init": 0.0008},
        {"hidden_layer_sizes": (80, 40, 20), "alpha": 0.001, "learning_rate_init": 0.0012},
    ]


def main():
    split_data = modelcommon.loadandsplit()
    scaled = modelcommon.scaledarrays(split_data)

    best_payload = None

    for run_number, settings in enumerate(buildruns(), start=1):
        model = MLPRegressor(
            hidden_layer_sizes=settings["hidden_layer_sizes"],
            alpha=settings["alpha"],
            learning_rate_init=settings["learning_rate_init"],
            activation="relu",
            solver="adam",
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=40,
            max_iter=4000,
            random_state=run_number * 11,
        )

        model.fit(scaled["x_train"], scaled["y_train"])

        scaled_predictions = model.predict(scaled["x_valid"])
        predictions = scaled["target_scaler"].inverse_transform(scaled_predictions.reshape(-1, 1)).ravel()
        metrics = modelcommon.metricsfrom(split_data["y_valid"], predictions)

        payload = {
            "model_name": "MLPRegressor",
            "run_number": run_number,
            "settings": settings,
            "metrics": metrics,
            "epochs": int(model.n_iter_),
        }

        if best_payload is None or payload["metrics"]["rmse"] < best_payload["metrics"]["rmse"]:
            best_payload = payload

    modelcommon.writeresult("mlprun_results.json", best_payload)
    modelcommon.printresult("MLPRegressor", best_payload)
    print(f"best settings: {best_payload['settings']}")


if __name__ == "__main__":
    main()
