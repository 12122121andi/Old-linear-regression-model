import math
import time

import numpy as np

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import modelcommon
import neutralnetworkcode as nn_module
ai = nn_module.ai


def scorebot(bot, x_values, y_values):
    total_error = 0.0
    for index in range(len(x_values)):
        prediction = ai.botcalc(bot, x_values[index])[0]
        difference = prediction - y_values[index]
        total_error += difference * difference
    mse = total_error / len(x_values)
    rmse = math.sqrt(mse)
    score = 1.0 / (1.0 + rmse)
    return score, rmse


def predictvalues(bot, x_values):
    outputs = []
    for row in x_values:
        outputs.append(ai.botcalc(bot, row)[0])
    return np.array(outputs, dtype=float)


def nextgeneration(elite_bots, elite_scores, input_count):
    bots = []

    for elite_bot in elite_bots:
        bots.append(elite_bot)

    for elite_bot, elite_score in zip(elite_bots, elite_scores):
        bots.extend(ai.basedbotcreator(elite_bot, elite_score, 18))

    bots.extend(ai.botcreator(input_count, 8, 1, 1, 24))
    return bots


def main():
    split_data = modelcommon.loadandsplit()
    scaled = modelcommon.scaledarrays(split_data)

    x_train = scaled["x_train"].tolist()
    x_valid = scaled["x_valid"].tolist()
    y_train = scaled["y_train"].tolist()

    input_count = len(split_data["feature_names"])
    bots = ai.botcreator(input_count, 8, 1, 1, 120)

    best_bot = None
    best_train_rmse = None
    best_score = None
    best_generation = 0
    minimum_delta = 0.001
    generation = 0
    last_improvement = time.time()
    train_started = time.time()

    while True:
        generation += 1
        scored_bots = []

        for bot in bots:
            score, train_rmse = scorebot(bot, x_train, y_train)
            scored_bots.append((score, train_rmse, bot))

        scored_bots.sort(key=lambda item: item[1])
        current_score, current_train_rmse, current_bot = scored_bots[0]

        if best_train_rmse is None or current_train_rmse < (best_train_rmse - minimum_delta):
            best_bot = current_bot
            best_train_rmse = current_train_rmse
            best_score = current_score
            best_generation = generation
            last_improvement = time.time()

        elite = scored_bots[:6]
        elite_bots = [item[2] for item in elite]
        elite_scores = [item[0] for item in elite]
        bots = nextgeneration(elite_bots, elite_scores, input_count)

        current_time = time.time()
        time_since_improvement = current_time - last_improvement
        total_time = current_time - train_started

        if time_since_improvement >= 120 and total_time >= 600:
            break

    scaled_predictions = predictvalues(best_bot, x_valid)
    predictions = scaled["target_scaler"].inverse_transform(scaled_predictions.reshape(-1, 1)).ravel()
    metrics = modelcommon.metricsfrom(split_data["y_valid"], predictions)

    payload = {
        "model_name": "CustomEvolutionNN",
        "metrics": metrics,
        "best_generation": best_generation,
        "best_scaled_train_rmse": float(best_train_rmse),
        "best_score": float(best_score),
        "seconds_without_improvement": 60,
        "total_training_seconds": float(time.time() - train_started),
        "network_shape": {
            "input_nodes": input_count,
            "hidden_nodes": 8,
            "hidden_layers": 1,
            "output_nodes": 1,
        },
    }

    modelcommon.writeresult("customNNrun_results.json", payload)
    modelcommon.printresult("CustomEvolutionNN", payload)
    print(f"best generation: {best_generation}")
    print(f"total training seconds: {payload['total_training_seconds']:.2f}")


if __name__ == "__main__":
    main()