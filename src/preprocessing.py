from typing import Literal

import numpy as np
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from transforms import FastTextTransform, PrepDataTransform


def f_trans(x):
    return x.squeeze()


def get_preprocessor(
    letter_count: bool,
    word_count: bool,
    lang: bool,
    scale: bool,
    stem: bool = False,
    text: Literal["count", "tfidf", "embed"] = "count",
):
    steps = [FunctionTransformer(f_trans)]
    if text == "embed":
        steps.append(FastTextTransform())
        if scale:
            steps.append(StandardScaler())
    else:
        steps.append(CountVectorizer(dtype=np.float64))
        if text == "tfidf":
            steps.append(TfidfTransformer())
        if scale:
            steps.append(StandardScaler(with_mean=False))
    post_content_transformer = make_pipeline(*steps)

    transformers = [(post_content_transformer, ["post_content"])]
    if lang:
        steps = [OneHotEncoder(handle_unknown="ignore")]
        if scale:
            steps.append(StandardScaler(with_mean=False))
        lang_transformer = make_pipeline(*steps)
        transformers.append((lang_transformer, ["lang"]))
    if scale:
        transformers.append(
            (StandardScaler(), make_column_selector(dtype_include=np.number))
        )

    preprocessor = make_column_transformer(*transformers, remainder="passthrough")
    preprocessor = make_pipeline(
        PrepDataTransform(stem, letter_count, word_count, lang), preprocessor
    )
    return preprocessor
