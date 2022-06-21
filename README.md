# social-media-post-classification
## Analysis
File `notebook_eda.ipynb` contains analysis of the dataset.

## Prediction
```commandline
cd src
python predict.py --input_filename file --output_filename file_out
```
All scripts should be run from the `src` directory.

## Data preparation
Additional features were introduced to improve performance of the model:
- lang - language of the post (one hot encoded)
- word_count - number of the words in the post (after removing new line symbols)
- letter_count - number of the letters in the post (after removing new line symbols)

Content of posts was stemmed using https://www.nltk.org/api/nltk.stem.snowball.html and stopwords (https://www.nltk.org/api/nltk.corpus.html) were removed. Additionally, [countvectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) with [TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) and fasttext embedding trained on the train data were used (separately).

During training [lgbm](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html#lightgbm.LGBMClassifier) and [mlp](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html) models were used with different hyperparameters, features and text preprocessing methods.

For each model and features the best set of hyperparameters was found with cross-validation on the training data. Next the best model was chosen based on the score on the validation data. In both situations `f1 score` was used. For the best model additional metrics were calculated and are saved in the `model/score.json` file.

After trails the best model was: lgbm (250 estimators, 100 leaves) with countvectorizer and tfidf (f1 score (val): 0.7821851078636046).

File `logs.txt` contains training logs.