import pandas as pd
import numpy as np
import sqlite3
from sqlalchemy import create_engine

messages = pd.read_csv('data/disaster_messages.csv')
categories = pd.read_csv('data/disaster_categories.csv')

df = pd.merge(messages, categories)


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

engine = create_engine('sqlite:///DisasterResponse.db')
df.to_sql('DisasterResponse', engine, index=False)