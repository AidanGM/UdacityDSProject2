# Udacity Data Science Project 2
## Disaster Response ML Pipeline

This project uses Figure Eight message data to train message classification system. This would be useful to filter our unimportant messages and direct messages to the right people. The trained model can be accessed through a provided web app, instructions will follow.

### Contents

[Packages](#Packages)

[Project Description](#Description)

[Files](#Files)

[Instructions](#Instructions)

[Additional Discussion](#Discussions)

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
- json

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

- `README.md`: Read me file (being read by you now)

- `data/process_data.py`: Python script to process data, containing ETL pipeline
- `data/DisasterResponse.db`: Disaster Response Database
- `data/disaster_categories.csv`: categories csv file
- `data/disaster_messages.csv`: messages csv file

- `models/train_classifier.py`: Python script to load data, create gridsearch to train model and store model
- `models/classifier.pkl` (EXCLUDED SINCE FILE IS TOO LARGE): Optimal model in wider parameter space 
- `models/classifier_balanced.pkl` (EXCLUDED SINCE FILE IS TOO LARGE): Optimal Random forest classifier with class frequency taken into account
- `models/classifier_balanced_low.pkl`: Optimal low-storage Random forest classifier with class frequency taken into account

- `app/templates/`: HTML Templates to be used for web app
- `app/run.py`: Python script to launch web app
  
## Instructions <a name="Instructions"></a>

To use the web app, download the `data`, `models` and `app` folders.

If you'd like to process the data, store this in a database and train a model run the following commands in the root directory:
1. `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
2. `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

You can then run the app by opening the apps directory and running the python script as follows:
1. `cd app`
2. `python run.py`

The current default uses the prebuilt classifier `models/classifier_balanced_low.pkl`, this is already trained and does not require you to do any data loading or model training.
Once the web app has been loaded you can try out any prompt! If you're not sure what to use, try this as your starting point:
"This is an emergency, I need help. Water is building up in my livingroom, I think it's because of the rain."

