import sys
import re
import numpy as np
import pandas as pd
import time
import pickle
from sqlalchemy import create_engine

import nltk
nltk.download(['punkt', 'wordnet'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier

def load_data(database_filepath):
    '''
    load data from database. Notice an extra step to replace all 2 in the category columns by 0.
    
    Input:
    database_filepath: string, the file path of database
    
    Output:
    X: the 'message' column of the database
    Y: the 'category' columns of the database
    category_names: list, containing the names of the category columns
    '''
    engine = create_engine('sqlite:///{}'.format(database_filepath)) 
    df = pd.read_sql_table('df', engine)
    X = df['message']
    Y = df[[col for col in df.columns if 'category' in col]]
    category_names = Y.columns
   
    return X, Y, category_names


def tokenize(text):
    '''
    tokenize the text
    
    Input:
    text: string, the original test
    
    Output:
    clean_tokens: list, the clean tokens
    '''
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''
    build a model and use grid search to find better parameters, with the following setups:
    pipeline: the machine learning pipeline with CountVectorizer, TF-IDF, and MultiOutputClassifier 
              using Randomforestclassifier
    parameters for grid search: n_estimators: [10, 20, 30],
                                min_samples_split: [2, 4, 6]
    
    Input:
    None
    
    Output:
    cv: from GridSearchCV
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(estimator=RandomForestClassifier()))
    ])
    
    parameters = {
            'clf__estimator__n_estimators': [20, 30],
            'clf__estimator__min_samples_split': [2, 4],
        }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    to evaluate the machine learning model using accuracy score and classification report
    
    Input:
    model: the machine learning model for fitting
    X_test: the X test dataset
    Y_test: the Y test dataset
    category_names: the names of categories
    
    Output:
    None, but best parameters, accuracy score and classification report will be printed
    '''
    Y_pred = model.predict(X_test)
    print("\nBest Parameters:", model.best_params_)
    print('After tuning, the accuracy score is ', accuracy_score(Y_test.values.flatten(), Y_pred.flatten()))
    print('\nThe classification report is \n', classification_report(Y_test.values[:,], Y_pred, target_names = Y_test.columns))


def save_model(model, model_filepath):
    '''
    to save model's best estimators in a pickle file
    
    Input:
    model: the machine learning model
    model_filepath: string, the desired filepath to save the model
    
    Output:
    None
    '''
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()