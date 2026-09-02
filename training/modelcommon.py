import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


csv_name = "Food_Delivery_Times.csv"
target_name = "Delivery_Time_min"
id_name = "Order_ID"
categorical_names = ["Weather", "Traffic_Level", "Time_of_Day", "Vehicle_Type"]
numeric_names = ["Distance_km", "Preparation_Time_min", "Courier_Experience_yrs"]
split_seed = 42


def csv_path():
    return Path(__file__).resolve().parent.parent / csv_name


def result_path(file_name):
    return Path(__file__).resolve().with_name(file_name)


def loadandsplit():
    data = pd.read_csv(csv_path())

    features = data.drop(columns=[target_name, id_name]).copy()
    targets = data[target_name].astype(float).copy()

    x_train, x_valid, y_train, y_valid = train_test_split(
        features,
        targets,
        test_size=0.2,
        random_state=split_seed,
    )

    numeric_fill = x_train[numeric_names].median()
    category_fill = {}
    for name in categorical_names:
        mode = x_train[name].mode(dropna=True)
        category_fill[name] = mode.iloc[0] if not mode.empty else "Missing"

    x_train = x_train.copy()
    x_valid = x_valid.copy()

    x_train[numeric_names] = x_train[numeric_names].fillna(numeric_fill)
    x_valid[numeric_names] = x_valid[numeric_names].fillna(numeric_fill)

    for name in categorical_names:
        x_train[name] = x_train[name].fillna(category_fill[name])
        x_valid[name] = x_valid[name].fillna(category_fill[name])

    x_train = pd.get_dummies(x_train, columns=categorical_names, dtype=float)
    x_valid = pd.get_dummies(x_valid, columns=categorical_names, dtype=float)

    x_train, x_valid = x_train.align(x_valid, join="left", axis=1, fill_value=0.0)
    x_train = x_train.sort_index(axis=1)
    x_valid = x_valid.reindex(columns=x_train.columns, fill_value=0.0)

    return {
        "x_train": x_train.astype(float),
        "x_valid": x_valid.astype(float),
        "y_train": y_train.reset_index(drop=True),
        "y_valid": y_valid.reset_index(drop=True),
        "feature_names": list(x_train.columns),
    }


def scaledarrays(split_data):
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()

    x_train = feature_scaler.fit_transform(split_data["x_train"])
    x_valid = feature_scaler.transform(split_data["x_valid"])

    y_train = target_scaler.fit_transform(split_data["y_train"].to_numpy().reshape(-1, 1)).ravel()
    y_valid = target_scaler.transform(split_data["y_valid"].to_numpy().reshape(-1, 1)).ravel()

    return {
        "x_train": x_train,
        "x_valid": x_valid,
        "y_train": y_train,
        "y_valid": y_valid,
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
    }


def metricsfrom(y_true, y_pred):
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
    }


def writeresult(result_file_name, payload):
    path = result_path(result_file_name)
    with path.open("w", encoding="utf-8") as result_file:
        json.dump(payload, result_file, indent=2)
    return path


def printresult(model_name, payload):
    metrics = payload["metrics"]
    print(model_name)
    print(f"validation rmse: {metrics['rmse']:.4f}")
    print(f"validation mae: {metrics['mae']:.4f}")
    print(f"validation r2: {metrics['r2']:.4f}")
