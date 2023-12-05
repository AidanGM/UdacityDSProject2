# import libraries
import sys

from sqlalchemy import create_engine

import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])

import re
import numpy as np
import pandas as pd

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

import pickle


def load_data(database_filepath:str):
    '''
    Input:
    database_filepath (str): file path to reach database
    
    Output:
    X (DataFrame): input data 
    Y (DataFrame): target data 
    category_names (list(str)): list containing column names  
    '''
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('MessageCategories',engine) # read in table

    X = df['message'] # messages
    Y = df.iloc[:,4:] # classification columns
    category_names = list(Y.columns) # classification column (category) names
    
    return X, Y, category_names

def tokenize(text:str) -> list:
    '''
    Input:
    text (str): raw text to be cleaned and tokenized
    
    Output
    clean_tokens (list(str)): list containing lemmatized tokens
    '''
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower()) # change to lower case, remove unimportant characters
    
    tokens = word_tokenize(text) # tokenize text
   
    sw = stopwords.words('english')
    lemmatizer = WordNetLemmatizer()
    clean_tokens = [lemmatizer.lemmatize(tok).strip() for tok in tokens if tok not in sw] # lemmatize and remove stopwords

    return clean_tokens


def build_model():
    '''
    Input:
    None
    
    Ouptut:
    cv: Gridsearch object containing the model and parameter space to search over
    '''
    pipeline = Pipeline([
        ('features', FeatureUnion([ # featureunion in pipeline structure to facilitate future additions

            ('text_pipeline', Pipeline([ # text processing pipeline
                ('vect', CountVectorizer(tokenizer=tokenize)), 
                ('tfidf', TfidfTransformer())
            ]))
        ])),

        ('moc', MultiOutputClassifier(estimator=RandomForestClassifier()))  # classifier
    ])
    
    # option for gridsearch with key identified hyperparameters with two different classifiers
    full_clf_gridsearch = [RandomForestClassifier(min_samples_split = mss, n_estimators = n_est, class_weight= cw , verbose=1)
                          for mss in (2,3) 
                          for n_est in (50,100) 
                          for cw in (None, "balanced")] + \
                          [AdaBoostClassifier(n_estimators = n_est) 
                          for n_est in (50,75)]
        
    # balanced random forest gridsearch (adds more weighting to class labels which appear less frequently) 
    balanced_rf_gridsearch = [RandomForestClassifier(min_samples_split = mss, class_weight= "balanced") for mss in (2,3)]
    
    low_storage = [RandomForestClassifier(n_estimators=50, class_weight="balanced")]
    
    parameters = {
        'features__text_pipeline__vect__ngram_range': [(1,1), (1,2)], #varying range of n_grams used
        'moc__estimator': low_storage
    }
    
    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=2) #scoring='f1' leads to errors

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Input: 
    model: Trained Multioutput classifier model
    X_test: Test X dataset
    Y_test: Test Y dataset
    category_names: names for each category in Y_test
    
    Output:
    displays classification report on predicted test values using the model
    '''
    Y_pred = model.predict(X_test)
                        
    for i, col in enumerate(category_names):
        print("Feature {}: {}".format(i+1,col))
        print(classification_report(Y_test[col],Y_pred[:,i]))
                         
    return True


def save_model(model, model_filepath):
    '''
    Input:
    model: Trained model
    model_filepath: filepath to save the model
    
    Output: 
    Save model in filepath as a pickle file
    '''
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)
    return True


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