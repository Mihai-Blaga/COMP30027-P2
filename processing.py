import pandas as pd


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
