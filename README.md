## Disaster-Response-Pipelines
# Introduction
This repository deals with disaster-related messages using ETL pipelines and NLP tools. It loads in data containing disaster-related messages, cleans the data, trains a machine learning model to classify the data into categories, then presents three data visualizations in a web app. This is a project of Udacity Data Scientist Nanodegree Program. 

# File structures
- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv  # data to process
|- process_data.py
|- DisasterResponse.db   # database to save clean data to

- models
|- train_classifier.py
|- classifier.pkl  # saved model, NOT UPLOADED TO GITHUB DUE TO FILE SIZE

- README.md

# Instruction
Please follow these steps to run the codes.
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. In another terminal window, run `env|grep WORK` to get the SPACEDOMAIN and SPACEID. Open a browser window and go to website https://SPACEID-3001.SPACEDOMAIN .
