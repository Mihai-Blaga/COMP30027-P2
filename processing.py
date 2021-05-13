import pandas as pd
import numpy as np


def create_csv_output(file_name: str, result: np.ndarray):
    output = pd.DataFrame({"duration_label": result})
    output.index += 1
    output.to_csv(file_name + ".csv", index_label="id")


# def preprocess_training(split=0, rs=None):
#     (X, y) = get_training(TRAIN_LOCATION)
#     X["n_ingredients"] = np.log(X["n_ingredients"] + 1)
#     X["n_steps"] = np.log(X["n_steps"] + 1)
#     X = X.loc[:, ["n_ingredients", "n_steps"]]

#     if split > 0:
#         X_train, X_test, y_train, y_test = train_test_split(
#             X, y, test_size=split, random_state=rs
#         )
#     else:
#         return (X, y)
#     scaler = StandardScaler().fit(X_train)
#     X_train = scaler.transform(X_train)
#     X_test = scaler.transform(X_test)

#     return (X_train, X_test, y_train, y_test)


# def preprocess_testing():
#     X = get_data(TEST_LOCATION)
#     X["n_ingredients"] = np.log(X["n_ingredients"] + 1)
#     X["n_steps"] = np.log(X["n_steps"] + 1)
#     X = X.loc[:, ["n_ingredients", "n_steps"]]
#     return X


def exclude_non_numeric(data: pd.DataFrame) -> pd.DataFrame:
    return data.loc[:, ["n_ingredients", "n_steps"]]


def combine_with_vec(
    X_train: pd.DataFrame,
    name_vec: pd.DataFrame,
    ingr_vec: pd.DataFrame,
    steps_vec: pd.DataFrame,
) -> pd.DataFrame:
    name_vec.columns = [str(label) + "_steps" for label in name_vec.columns]
    ingr_vec.columns = [str(label) + "_ingrs" for label in ingr_vec.columns]
    steps_vec.columns = [str(label) + "_name" for label in steps_vec.columns]

    combined: pd.DataFrame = pd.concat(
        [X_train, name_vec, ingr_vec, steps_vec], axis=1, join="inner"
    )

    return combined
