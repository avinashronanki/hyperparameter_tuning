import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from tpot import TPOTRegressor

concrete = pd.read_excel("Concrete.xlsx")
# concrete.head()
#concrete.info()
# concrete.describe()

X = concrete.drop("Comp_str", axis = 1)
y = concrete["Comp_str"]

# print(X.shape)
# print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42) #spliting the data 

# scale the variables
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf = GradientBoostingRegressor(criterion = "mae")
clf.get_params()

# Create a Gradient Boosted Regressor with specified criterion
gb_regressor = GradientBoostingRegressor(criterion = "mae")
# Create the parameter grid
param_grid = {'max_depth' : [2, 4, 8, 10, 12],
              'n_estimators' : [100, 200, 300],
              'max_features' : ['auto', 'sqrt'],
              "criterion" : ["friedman_mse", "mse", "mae"]}
# Create a GridSearchCV object
grid_gb = GridSearchCV(
    estimator = gb_regressor,
    param_grid = param_grid,
    scoring = 'neg_mean_absolute_error',
    n_jobs = 4,
    cv = 10,
    refit = True,
    return_train_score = True)
# print(grid_gb)
#model trainig 

grid_gb.fit(X_train_scaled, y_train)