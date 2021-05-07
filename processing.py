import scipy.stats
import scipy
import pickle
import pandas as pd
import numpy as np

from scipy.sparse.csr import csr_matrix

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer

from typing import Tuple

TEST_LOCATION = "./COMP30027_2021_Project2_datasets/recipe_test.csv"
TRAIN_LOCATION = "./COMP30027_2021_Project2_datasets/recipe_train.csv"

# Loading in the datasets within the zip files based on steps in README
BASE_LOCATION = "./COMP30027_2021_Project2_datasets/"
COUNT_VEC_LOCATION = BASE_LOCATION + "recipe_text_features_countvec/"
DOC2VEC50_VEC_LOCATION = BASE_LOCATION + "recipe_text_features_doc2vec50/"
DOC2VEC100_VEC_LOCATION = BASE_LOCATION + "recipe_text_features_doc2vec100/"


def get_data(csv_location: str) -> pd.DataFrame:
    # Change 9999 to np.nan when reading in the data
    data = pd.read_csv(csv_location, header=0)
    return data


def create_csv_output(file_name: str, result: np.ndarray):
    output = pd.DataFrame({"duration_label": result})
    output.index += 1
    output.to_csv(file_name + ".csv", index_label="id")


def get_training(train_loc: str):
    train = get_data(train_loc)
    X = train.iloc[:, :-1]
    y = train.iloc[:, -1]
    return (X, y)


def preprocess_training(split=0, rs=None):
    (X, y) = get_training(TRAIN_LOCATION)
    X["n_ingredients"] = np.log(X["n_ingredients"])
    X["n_steps"] = np.log(X["n_steps"])
    X = X.loc[:, ["n_ingredients", "n_steps"]]

    if split > 0:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=split, random_state=rs
        )
    else:
        return (X, y)

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    return (X_train, X_test, y_train, y_test)


def preprocess_testing():
    X = get_data(TEST_LOCATION)
    X["n_ingredients"] = np.log(X["n_ingredients"])
    X["n_steps"] = np.log(X["n_steps"])
    X = X.loc[:, ["n_ingredients", "n_steps"]]
    return X


def get_countvec() -> Tuple[CountVectorizer, CountVectorizer, CountVectorizer]:
    """
    return CountVec in order of name, ingr and steps.
    """
    return (
        pickle.load(open(COUNT_VEC_LOCATION + "train_name_countvectorizer.pkl", "rb")),
        pickle.load(open(COUNT_VEC_LOCATION + "train_ingr_countvectorizer.pkl", "rb")),
        pickle.load(open(COUNT_VEC_LOCATION + "train_steps_countvectorizer.pkl", "rb")),
    )


def get_sparse(data: str = "train") -> Tuple[csr_matrix, csr_matrix, csr_matrix]:
    """
    column = word occurence, row = individual instance
    set data = "train" for train dataset and "test" for test dataset
    Return Sparse matrix in order of name, ingr, steps.
    """
    if data == "train":
        return (
            scipy.sparse.load_npz(COUNT_VEC_LOCATION + "train_name_vec.npz"),
            scipy.sparse.load_npz(COUNT_VEC_LOCATION + "train_ingr_vec.npz"),
            scipy.sparse.load_npz(COUNT_VEC_LOCATION + "train_steps_vec.npz"),
        )
    elif data == "test":
        return (
            scipy.sparse.load_npz(COUNT_VEC_LOCATION + "test_name_vec.npz"),
            scipy.sparse.load_npz(COUNT_VEC_LOCATION + "test_ingr_vec.npz"),
            scipy.sparse.load_npz(COUNT_VEC_LOCATION + "test_steps_vec.npz"),
        )
    else:
        return None


def get_Doc2Vec(data: str = "train", num_features: int = 50):
    """
    Doc2Vec representation with 50/100 features.
    Think of them as 50 dimensional vectors representing the words/phrases
    If data == "train" return train dataset and if data == "test" return test dataset.
    Specify 50 or 100 features with num_features.
    Return 3 DataFrames in order of name, ingr and steps.
    """
    if data == "train":
        if num_features == 50:
            return (
                pd.read_csv(
                    DOC2VEC50_VEC_LOCATION + "train_name_doc2vec50.csv",
                    index_col=False,
                    delimiter=",",
                    header=None,
                ),
                pd.read_csv(
                    DOC2VEC50_VEC_LOCATION + "train_ingr_doc2vec50.csv",
                    index_col=False,
                    delimiter=",",
                    header=None,
                ),
                pd.read_csv(
                    DOC2VEC50_VEC_LOCATION + "train_steps_doc2vec50.csv",
                    index_col=False,
                    delimiter=",",
                    header=None,
                ),
            )
        elif num_features == 100:
            return (
                pd.read_csv(
                    DOC2VEC100_VEC_LOCATION + "train_name_doc2vec100.csv",
                    index_col=False,
                    delimiter=",",
                    header=None,
                ),
                pd.read_csv(
                    DOC2VEC100_VEC_LOCATION + "train_ingr_doc2vec100.csv",
                    index_col=False,
                    delimiter=",",
                    header=None,
                ),
                pd.read_csv(
                    DOC2VEC100_VEC_LOCATION + "train_steps_doc2vec100.csv",
                    index_col=False,
                    delimiter=",",
                    header=None,
                ),
            )
        else:
            return None
    elif data == "test":
        if num_features == 50:
            return (
                pd.read_csv(
                    DOC2VEC50_VEC_LOCATION + "test_name_doc2vec50.csv",
                    index_col=False,
                    delimiter=",",
                    header=None,
                ),
                pd.read_csv(
                    DOC2VEC50_VEC_LOCATION + "test_ingr_doc2vec50.csv",
                    index_col=False,
                    delimiter=",",
                    header=None,
                ),
                pd.read_csv(
                    DOC2VEC50_VEC_LOCATION + "test_steps_doc2vec50.csv",
                    index_col=False,
                    delimiter=",",
                    header=None,
                ),
            )
        elif num_features == 100:
            return (
                pd.read_csv(
                    DOC2VEC100_VEC_LOCATION + "test_name_doc2vec100.csv",
                    index_col=False,
                    delimiter=",",
                    header=None,
                ),
                pd.read_csv(
                    DOC2VEC100_VEC_LOCATION + "test_ingr_doc2vec100.csv",
                    index_col=False,
                    delimiter=",",
                    header=None,
                ),
                pd.read_csv(
                    DOC2VEC100_VEC_LOCATION + "test_steps_doc2vec100.csv",
                    index_col=False,
                    delimiter=",",
                    header=None,
                ),
            )
        else:
            return None
    else:
        return None
