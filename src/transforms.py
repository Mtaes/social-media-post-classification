import re
import string

import nltk
import numpy as np
import pandas as pd
from gensim.models.fasttext import FastText
from iso639 import languages
from langdetect import DetectorFactory, LangDetectException, detect
from nltk.stem import SnowballStemmer
from sklearn.base import BaseEstimator


def detect_lang(txt):
    DetectorFactory.seed = 42
    try:
        lang = detect(txt)
    except LangDetectException:
        lang = "unknown"
    return lang


def prep_df(df: pd.DataFrame):
    pd.set_option("mode.chained_assignment", None)
    df["post_content"] = df["post_content"].apply(
        lambda x: re.sub("\n", " ", x.lower())
    )
    df["letter_count"] = df["post_content"].apply(lambda x: len(x))
    df["word_count"] = df["post_content"].apply(lambda x: len(x.split()))
    df["post_content"] = df["post_content"].apply(
        lambda x: re.sub(f"[{re.escape(string.punctuation)}]", " ", x)
    )
    df["lang"] = df["post_content"].apply(detect_lang)
    df["post_content_stem"] = df["post_content"].copy()

    for lang in df["lang"].unique():
        try:
            lang_name = languages.get(alpha2=lang).name.lower()
        except KeyError:
            lang_name = None
        if lang_name is not None:
            indexes = df["lang"] == lang
            try:
                stopwords = nltk.corpus.stopwords.words(lang_name.lower())
            except OSError:
                stopwords = None
            if stopwords is not None:
                stopwords = (stopword.lower() for stopword in stopwords)
                df.loc[indexes, "post_content"] = df.loc[indexes, "post_content"].apply(
                    lambda x: " ".join(
                        [word for word in x.split() if word not in stopwords]
                    )
                )
                df.loc[indexes, "post_content_stem"] = df.loc[
                    indexes, "post_content"
                ].copy()
            if lang_name in SnowballStemmer.languages:
                stemmer = SnowballStemmer(lang_name)
                df.loc[indexes, "post_content_stem"] = df.loc[
                    indexes, "post_content"
                ].apply(lambda x: " ".join([stemmer.stem(word) for word in x.split()]))
    return df


class PrepDataTransform:
    def __init__(self, stem: bool, letter_count: bool, word_count: bool, lang: bool):
        self.stem = stem
        self.letter_count = letter_count
        self.word_count = word_count
        self.lang = lang

    def fit(self, x, y=None, sample_weight=None):
        return self

    def transform(self, x, copy=None):
        new_df = pd.DataFrame()
        if self.stem:
            new_df["post_content"] = x["post_content_stem"].copy()
        else:
            new_df["post_content"] = x["post_content"].copy()
        if self.letter_count:
            new_df["letter_count"] = x["letter_count"].copy()
        if self.word_count:
            new_df["word_count"] = x["word_count"].copy()
        if self.lang:
            new_df["lang"] = x["lang"].copy()
        return new_df


class FastTextTransform(BaseEstimator):
    def __init__(self, vector_size=300, epochs=5):
        self.vector_size = vector_size
        self.epochs = epochs
        self.embed = FastText(vector_size=vector_size, epochs=epochs)

    def fit(self, x, y=None, sample_weight=None):
        tokens = [x.split() for x in x.tolist()]
        self.embed = FastText(tokens, vector_size=self.vector_size, epochs=self.epochs)
        return self

    def compute_doc_embed(self, x):
        x = x.split()
        embed = []
        for token in x:
            embed.append(self.embed.wv[token])
        embed = sum(embed) / max(len(x), 1)
        if type(embed) is float:
            embed = np.full((self.vector_size,), embed, dtype=np.float)
        return embed

    def transform(self, x, copy=None):
        result = x.apply(self.compute_doc_embed)
        new_x = []
        for i in range(self.vector_size):
            tmp = []
            for res in result.to_list():
                tmp.append(res[i])
            new_x.append(tmp)
        new_x = pd.DataFrame(new_x).T
        return new_x
