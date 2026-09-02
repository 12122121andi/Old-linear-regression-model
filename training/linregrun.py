from sklearn.linear_model import LinearRegression

import modelcommon


def main():
    split_data = modelcommon.loadandsplit()

    model = LinearRegression()
    model.fit(split_data["x_train"], split_data["y_train"])

    predictions = model.predict(split_data["x_valid"])
    metrics = modelcommon.metricsfrom(split_data["y_valid"], predictions)

    payload = {
        "model_name": "LinearRegression",
        "metrics": metrics,
        "feature_count": len(split_data["feature_names"]),
    }

    modelcommon.writeresult("linregrun_results.json", payload)
    modelcommon.printresult("LinearRegression", payload)


if __name__ == "__main__":
    main()
