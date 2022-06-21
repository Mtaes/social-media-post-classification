import json
from pathlib import Path

import pandas as pd
from joblib import dump
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline

from preprocessing import get_preprocessor
from transforms import prep_df


def load_data(path_to_train: str, path_to_val_x: str, path_to_val_y: str):
    train_df = pd.read_parquet(path_to_train)
    val_df = pd.read_parquet(path_to_val_x)
    val_df["platform"] = pd.read_parquet(path_to_val_y)["platform"]
    return train_df, val_df


def train_model(
    model, params_grid: dict, info: dict, train_df: pd.DataFrame, val_df: pd.DataFrame
):
    preprocessor = get_preprocessor(
        letter_count=info["letter_count"],
        word_count=info["word_count"],
        lang=info["lang"],
        scale=info["scale"],
        stem=info["stem"],
        text=info["text"],
    )
    clf = make_pipeline(preprocessor, model)
    clf = GridSearchCV(clf, params_grid, verbose=3)
    clf.fit(train_df, train_df["platform"])
    best_params = clf.best_params_
    best_model = clf.best_estimator_
    preds = best_model.predict(val_df)
    score = {
        "f1": f1_score(val_df["platform"], preds),
        "report": classification_report(val_df["platform"], preds, output_dict=True),
    }
    return best_model, best_params, score


def save_model(model_path: Path, model, params, score, info):
    dump(model, model_path / "model.joblib", compress=9)
    with open(model_path / "params.json", "w") as f:
        json.dump(params, f, indent=4)
    with open(model_path / "score.json", "w") as f:
        json.dump(score, f, indent=4)
    with open(model_path / "info.json", "w") as f:
        json.dump(info, f, indent=4)


def train_models(train_df: pd.DataFrame, val_df: pd.DataFrame):
    train_df = prep_df(train_df)
    val_df = prep_df(val_df)
    train_df = train_df["platform"].replace({"FB": 0, "TW": 1})
    val_df = val_df["platform"].replace({"FB": 0, "TW": 1})
    model_path = Path("..") / "model"
    model_path.mkdir(parents=True, exist_ok=True)

    # Train basic model
    info = {
        "letter_count": False,
        "word_count": False,
        "lang": False,
        "scale": False,
        "stem": True,
        "text": "count",
    }
    params_grid = {
        "lgbmclassifier__num_leaves": (31, 100),
        "lgbmclassifier__n_estimators": (100, 250),
        "pipeline__columntransformer__pipeline__countvectorizer__ngram_range": (
            (1, 1),
            (1, 3),
        ),
    }
    model, params, score = train_model(
        model=LGBMClassifier(),
        params_grid=params_grid,
        info=info,
        train_df=train_df,
        val_df=val_df,
    )
    best_score = score
    best_params = params
    best_info = info
    save_model(model_path, model, params, score, info)

    # Change text preprocessing
    info = {
        "letter_count": False,
        "word_count": False,
        "lang": False,
        "scale": False,
        "stem": True,
        "text": "tfidf",
    }
    params_grid = {
        "lgbmclassifier__num_leaves": (31, 100),
        "lgbmclassifier__n_estimators": (100, 250),
        "pipeline__columntransformer__pipeline__countvectorizer__ngram_range": (
            (1, 1),
            (1, 3),
        ),
    }
    model, params, score = train_model(
        model=LGBMClassifier(),
        params_grid=params_grid,
        info=info,
        train_df=train_df,
        val_df=val_df,
    )
    if score["f1"] > best_score["f1"]:
        best_score = score
        best_params = params
        best_info = info
        save_model(model_path, model, params, score, info)
    best_text = best_info["text"]

    # Add features
    info = {
        "letter_count": True,
        "word_count": True,
        "lang": True,
        "scale": False,
        "stem": True,
        "text": best_text,
    }
    params_grid = {
        "lgbmclassifier__num_leaves": (31, 100),
        "lgbmclassifier__n_estimators": (100, 250),
        "pipeline__columntransformer__pipeline-1__countvectorizer__ngram_range": (
            (1, 1),
            (1, 3),
        ),
    }
    model, params, score = train_model(
        model=LGBMClassifier(),
        params_grid=params_grid,
        info=info,
        train_df=train_df,
        val_df=val_df,
    )
    if score["f1"] > best_score["f1"]:
        best_score = score
        best_params = params
        best_info = info
        save_model(model_path, model, params, score, info)

    add_features = best_info["lang"]

    # Add fasttext
    info = {
        "letter_count": add_features,
        "word_count": add_features,
        "lang": add_features,
        "scale": False,
        "stem": True,
        "text": "embed",
    }
    params_grid = {
        "lgbmclassifier__num_leaves": (31, 100),
        "lgbmclassifier__n_estimators": (100, 250),
        f"pipeline__columntransformer__pipeline{'-1' if add_features else ''}__fasttexttransform__vector_size": (
            100,
        ),
        f"pipeline__columntransformer__pipeline{'-1' if add_features else ''}__fasttexttransform__epochs": (
            10,
        ),
    }
    model, params, score = train_model(
        model=LGBMClassifier(),
        params_grid=params_grid,
        info=info,
        train_df=train_df,
        val_df=val_df,
    )
    if score["f1"] > best_score["f1"]:
        best_score = score
        best_params = params
        best_info = info
        save_model(model_path, model, params, score, info)

    # Use mlp
    info = {
        "letter_count": add_features,
        "word_count": add_features,
        "lang": add_features,
        "scale": True,
        "stem": True,
        "text": "embed",
    }
    params_grid = {
        "mlpclassifier__hidden_layer_sizes": ((50,),),
        f"pipeline__columntransformer__pipeline{'-1' if add_features else ''}__fasttexttransform__vector_size": (
            100,
        ),
        f"pipeline__columntransformer__pipeline{'-1' if add_features else ''}__fasttexttransform__epochs": (
            10,
        ),
    }
    model, params, score = train_model(
        model=MLPClassifier(max_iter=1000),
        params_grid=params_grid,
        info=info,
        train_df=train_df,
        val_df=val_df,
    )
    if score["f1"] > best_score["f1"]:
        best_score = score
        best_params = params
        best_info = info
        save_model(model_path, model, params, score, info)

    print(best_score)
    print(best_params)
    print(best_info)
