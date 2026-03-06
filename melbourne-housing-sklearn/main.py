import time

from sklearn.linear_model import LinearRegression

import numpy as np
import os
from zlib import crc32
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.feature_selection import SelectPercentile, f_regression
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.model_selection import cross_validate
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.preprocessing import FunctionTransformer
from sklearn.tree import DecisionTreeRegressor

to_dense = FunctionTransformer(
    lambda X: X.toarray() if hasattr(X, "toarray") else X,
    accept_sparse=True,
    feature_names_out="one-to-one"
)



HOUSING_PATH="MELBOURNE_HOUSE_PRICES_LESS.csv"
def load_housing_data(housing_path=HOUSING_PATH):
    return pd.read_csv(HOUSING_PATH)

from sklearn.base import BaseEstimator, TransformerMixin

class NewAttributes(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        Z = X.copy()
        s = (Z["Postcode"].astype(str)
                          .str.slice(0, 2))
        Z["postcode_area"] = s
        Z = Z.drop(columns=["Address"])
        return Z
housing = load_housing_data(HOUSING_PATH)
housing = housing[housing["Price"].notna()].copy()
target="Price"
Y=housing[target]
X=housing.drop(columns=target)
X["Postcode"] = X["Postcode"].astype("Int64").astype(str)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

imputer = SimpleImputer(strategy="median")

num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("power", PowerTransformer(method="yeo-johnson")),
    ("scaler", StandardScaler())
])

cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=0.02))
])

preprocess = Pipeline([
    ("add_attrs", NewAttributes()),
    ("ct", ColumnTransformer([
        ("num", num_pipe, selector(dtype_include="number")),
        ("cat", cat_pipe, selector(dtype_exclude="number")),
    ]))
])

Xt_train = preprocess.fit_transform(X_train, y_train)
Xt_test  = preprocess.transform(X_test)

num_cols = X_train.select_dtypes(include="number").columns.tolist()
show_cols = ["Distance","Propertycount","Rooms"]

X_train[show_cols].hist(bins=30, figsize=(8,3))
plt.suptitle("PRZED")
plt.tight_layout()
plt.show()

num_pipe_fit = num_pipe.fit(X_train[show_cols])
Xt_num = num_pipe_fit.transform(X_train[show_cols])
pd.DataFrame(Xt_num, columns=[f"t_{c}" for c in show_cols]).hist(bins=30, figsize=(8,3))
plt.suptitle("PO"); plt.tight_layout(); plt.show()

sel = SelectPercentile(score_func=f_regression, percentile=80)

def build_pipe(model, with_selection: bool):
    steps = [("prep", preprocess)]
    if with_selection:
        steps.append(("sel", sel))
    if isinstance(model, LinearRegression):
        steps.append(("to_dense", to_dense))
    steps.append(("model", model))
    return Pipeline(steps)
models = {
    "lin": LinearRegression(),
    "dt":  DecisionTreeRegressor(random_state=42, max_depth=8),
    "rf":  RandomForestRegressor(random_state=42, n_jobs=-1),
}

def run(with_selection: bool):
    print(f"\n\n===== run(with_selection={with_selection}) =====")
    t0 = time.perf_counter()
    cv_rows = {}
    for name, mdl in models.items():
        pipe=build_pipe(mdl, with_selection)

        scores = cross_validate(
            pipe, X_train, y_train, cv=5, n_jobs=-1,
            scoring={"rmse": "neg_root_mean_squared_error",
                     "mae": "neg_mean_absolute_error"}
        )
        cv_rows[name] = {
            "RMSE": -scores["test_rmse"].mean(),
            "MAE": -scores["test_mae"].mean(),
        }

    cv_df = pd.DataFrame(cv_rows).T.sort_values("RMSE")
    print("\n=== CV wyniki ===\n", cv_df)
    best_name = cv_df.index[0]
    print("Najlepszy model z CV:", best_name)

    params = {
        "lin": (LinearRegression(), {}),
        "dt": (
            DecisionTreeRegressor(random_state=42),
            {
                "model__max_depth": randint(2, 20),
                "model__min_samples_leaf": randint(1, 20),
                "model__min_samples_split": randint(2, 20),
            }
        ),
        "rf": (
            RandomForestRegressor(random_state=42, n_jobs=1, max_samples=0.7),
            {
                "model__n_estimators": randint(100, 300),
                "model__max_depth": randint(3, 20),
                "model__min_samples_leaf": randint(1, 10),
                "model__min_samples_split": randint(2, 20),
                "model__max_features": uniform(0.3, 0.5),
            }
        ),
    }
    base_est, param_dist = params[best_name]
    pipe_best = build_pipe(base_est, with_selection)

    if param_dist:
        search = RandomizedSearchCV(
            pipe_best, param_distributions=param_dist,
            n_iter=15, cv=3, n_jobs=-1,
            scoring="neg_root_mean_squared_error",
            random_state=42, verbose=2
        )
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        print("Best params:", search.best_params_)
        print("Best CV RMSE:", -search.best_score_)
    else:
        best_model = pipe_best.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    print("\n=== Test ===")
    print({
        "Test_RMSE": rmse,
        "Test_MAE": mean_absolute_error(y_test, y_pred),
    })
    t1 = time.perf_counter()
    print(f"Total time run(with_selection={with_selection}): {t1 - t0:.2f} s")





run(False)
run(True)

