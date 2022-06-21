from argparse import ArgumentParser

import nltk

from utils import load_data, train_models

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--path_to_train", type=str)
    parser.add_argument("--path_to_val_x", type=str)
    parser.add_argument("--path_to_val_y", type=str)
    args = parser.parse_args()
    nltk.download("stopwords")
    train_df, val_df = load_data(
        path_to_train=args.path_to_train,
        path_to_val_x=args.path_to_val_x,
        path_to_val_y=args.path_to_val_y,
    )
    train_models(train_df, val_df)
