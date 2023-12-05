# Udacity Data Science Project 2
## Disaster Response ML Pipeline

### Contents

[Packages](#Packages)

[Project Description](#Description)

[Files](#Files)

[Instructions](#Instructions)

## Packages <a name="Packages"></a>

The following libraries were used in this project

- sys: used in setup
- pandas
- sqlalchemy: to store and load databases
- nltk: NLP
- re: NLP 
- numpy: Used in machine learning
- sklearn: Main machine learning module
- pickle: Storing models 
- Flask: Developing web app
- plotly: Creating plots

## Project Description <a name="Description"></a>

This project is part of the Udacity Data Science Nanodegree. The goal is to use classified message data provided by Figure 8 to produce a full machine learning pipeline. This includes

An **ETL Pipeline** (`data/process_data.py`) which
- extracts data from `messages` and `categories` datasets
- transforms the data by merging and cleaning the data
- loads the data into an SQLite database

A **ML Pipeline** (`models/train_classifier.py`) which

- loads data
- splits into train and test sets
- builds a pipeline consisting of text-related features (TF-IDF) and a classifier
- trains a model and grid searches over a reasonable parameter space
- stores model in a pickle file (The main model is not included in the files as it is too large for github, instead `models/classifier_balanced_low` is included)

A **Web App** (`app`) which can take text input and displays the models resulting message categories.

## Files <a name="Files"></a>


## Instructions <a name="Instructions"></a>


## Additional Discussions <a name="discussions"</a>
