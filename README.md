# Old Linear Regression Model

This project started on 3/17/2023, when I was 16 and first learned about how machine learning could be used to solve problems impossible for classical programming. I already knew python, but I had no idea that libraries like pytorch or scikit-learn existed, so I jumped straight into building a neural network from scratch. My results are in `neuralnetworkcode.py`.

When reviewing the code today, I can tell 16 year old me didn't really understand how neural networks worked. For starters, there is no activation function anywhere in the code. Basically meaning that all of this complexity and structure basically amounted to a fancy linear regression model. Furthermore, there is no backpropagation at all (rather cleverly). Instead of gradient descent, the network is trained with mutations. A population of bots (which are just sets of weights) is generated, and each generation mutates the previous best bot by an amount that scales inversely with how accurate it was, and the process repeats indefinitly. Yep, indefinitly, because there is also not a preprogramed method of stopping, meaning the amount of generations was predetermined before training. I didn't know anything about activation functions or gradient descent, so I built the only thing that I could think of at the time: evolution by natural selection. 

This repo now compares that original custom network against three standard baselines on `Food_Delivery_Times.csv`:

- `linregrun.py`
- `xgboostrun.py`
- `mlprun.py`
- `customNNrun.py`

## Data setup

All four scripts use the same preprocessing and the same 80/20 train-validation split.

- `Order_ID` is dropped because it is only an identifier.
- Numeric missing values are filled from the training split median.
- Categorical missing values are filled from the training split mode.
- Categorical columns are converted into binary one-hot features.

The categorical columns are:

- `Weather`
- `Traffic_Level`
- `Time_of_Day`
- `Vehicle_Type`

That means a column like `Traffic_Level` becomes separate binary columns such as `Traffic_Level_High`, `Traffic_Level_Low`, and `Traffic_Level_Medium`.

- For my model, since there is no preprogrammed stop, I had to force the training program to run for at least 10 minutes and train until 120s of no improvement.

## Validation results

| Model | RMSE | MAE | R2 | Notes |
| --- | ---: | ---: | ---: | --- |
| Linear Regression | 8.9522 | 6.0634 | 0.8212 | Best score on this split |
| XGBoost | 9.1376 | 6.5410 | 0.8137 | Best tuned run used depth 4 and early stopping |
| MLP Regressor | 9.5578 | 6.8524 | 0.7962 | Best tuned run used hidden layers `(80, 40, 20)` |
| Custom Evolution NN | 10.2191 | 7.4457 | 0.7670 | Trained until 120s stagnation and 10m minimum runtime |

## What this says about my model

The strongest result was linear regression, showing that the data is mostly linear. Furthermore, the fact that more complex architectures all increased loss shows that added model complexity was actively unhelpful rather than netural. 

My custom network did learn useful signal, however it lagged behind every library model, for a few reasons:

- No activation function: `botcalc` never applies a non-linearity between layers, only weights and biases. This meant this was mathematically identical to a single linear layer, only taking a much longer time due to added complexity
- No gradient-based optimization: Instead of backpropagation, which is what MLP uses, the network is trained by random mutation and keeping the best performing generation's bot. This is a usable strategy, but is much less efficent than backprop. 
- C-optimized libraries: scikit-learn and XGBoost are backed by vectorized, C-optimized numerical code. Being in a C++ optimization class now, I am well aware of the performance gap between low level C and high level python. Since my implementation is pure python, that explains the 636 second training time.

## If I rebuilt this today

With what I know now, I would just use MLP instead of trying to build a Neural Network from scratch.

But if I had to improve this system, I would simply add a activation function like ReLU between laters, and replace my mutation strategy with an optimizer like Adam. Lastly, I would preform calculations with Pandas rather then with nested python loops. 

These changes would increase the preformance of my model, especially on complex non linear data, and increase the speed at least 600 fold.