from argparse import ArgumentParser
from pathlib import Path

import nltk
import pandas as pd
from joblib import load

from transforms import prep_df

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input_filename", type=Path)
    parser.add_argument("--output_filename", type=Path)
    args = parser.parse_args()
    nltk.download("stopwords")
    clf = load("../model/model.joblib")
    test_df = pd.read_parquet(args.input_filename)
    test_df = prep_df(test_df)
    predictions = clf.predict(test_df)
    predictions = pd.DataFrame({"platform": predictions})
    predictions["platform"] = predictions["platform"].replace({0: "FB", 1: "TW"})
    predictions.to_parquet(args.output_filename)
