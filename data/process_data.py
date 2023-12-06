# importing libraries
import sys

import pandas as pd

from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    Input:
    messages_filepath (str): filepath to get to messages dataset
    categories_filepath (str): filepath to get to categories dataset
    
    Output:
    df (DataFrame): DataFrame containing the inner joined messages and categories
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    df = pd.merge(messages, categories, left_on='id', right_on='id')
    
    return df

def clean_data(df):
    '''
    Input: 
    df (DataFrame): DataFrame of messages and categories dataset
    
    Output:
    df (DataFrame): Dataframe of cleaned messages and categories data
    '''
    # converting category data from list separated by ';' to a dummy variable for each category
    categories = df['categories'].str.split(pat=';', expand=True)
    row = categories.iloc[0]
    category_colnames = list(row.apply(lambda x: x[:-2]))
    categories.columns = category_colnames
    
    for column in categories:
        categories[column] = categories[column].astype(str).str[-1]
        categories[column] = categories[column].astype(int)
        
    
    
    
    df = df.drop(['categories'], axis=1)

    df = pd.concat([df, categories], axis=1)
    
    # dropping rows where related is 2
    df = df[df['related'] != 2]
    
    # removing duplicates
    df = df.drop_duplicates()
    
    return df


def save_data(df, database_filename):
    '''
    Input: 
    df (DataFrame): DataFrame with message and category dataset
    database_filename (str): File name for database to store data in
    
    Output:
    saves data in database under "MessageCategories"
    '''
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('MessageCategories', engine, index=False, if_exists='replace') 


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
