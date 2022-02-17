import sys
import pandas as pd
import numpy as np
import sqlite3
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    load messages and categories data and merge them together in a dataframe
    
    Input: 
    messages_filepath: string, file path of messages data
    categories_filepath: string, file path of messages data
    
    Output:
    df: dataframe, containing messages and categories data
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories)
    return df

    
def clean_data(df):
    '''
    to clean the dataframe, including splitting the categories columns and dropping duplicates
    
    Input:
    df: dataframe containing messages and categories data
    
    Output:
    df: cleaned dataframe
    '''
    categories = df['categories'].str.split(';', expand=True) # create a dataframe of the 36 individual category columns
    row = categories.iloc[0] # select the first row of the categories dataframe
    category_colnames = row.str.slice(stop=-2) # extract a list of new column names for categories.
    categories.columns = 'category_' + category_colnames # rename the columns of `categories`

    for column in categories:
        categories[column] = categories[column].str.slice(start=-1) # set each value to be the last character of the string
        categories[column] = categories[column].astype(int) # convert column from string to numeric

    try: # drop the original categories column from `df`
        df.drop(columns=['categories'], axis=1, inplace=True)
    except:
        pass

    df = pd.concat([df, categories], axis=1) # concatenate the original dataframe with the new `categories` dataframe
    df.drop_duplicates(inplace=True)
    return df

def save_data(df, database_filepath):
    '''
    to save data using sqlite
    
    Input:
    df: dataframe to be saved
    database_filepath: the file path where the database will be saved
    
    Output: 
    None
    '''
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df.to_sql(database_filepath, engine, index=False)


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