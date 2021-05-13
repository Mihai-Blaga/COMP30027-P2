import pandas as pd
import pickle
import scipy


from typing import Tuple

from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse.csr import csr_matrix


TEST_LOCATION = "./COMP30027_2021_Project2_datasets/recipe_test.csv"
TRAIN_LOCATION = "./COMP30027_2021_Project2_datasets/recipe_train.csv"

# Loading in the datasets within the zip files based on steps in README
BASE_LOCATION = "./COMP30027_2021_Project2_datasets/"
COUNT_VEC_LOCATION = BASE_LOCATION + "recipe_text_features_countvec/"
DOC2VEC50_VEC_LOCATION = BASE_LOCATION + "recipe_text_features_doc2vec50/"
DOC2VEC100_VEC_LOCATION = BASE_LOCATION + "recipe_text_features_doc2vec100/"


def get_data(csv_location: str) -> pd.DataFrame:
    data = pd.read_csv(csv_location, header=0)
    return data


def get_test(test_loc: str = TEST_LOCATION):
    return get_data(test_loc)


def get_training(train_loc: str = TRAIN_LOCATION):
    train = get_data(train_loc)
    X = train.iloc[:, :-1]
    y = train.iloc[:, -1]
    return (X, y)


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
