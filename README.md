# Recommender System

## Overview

This project simulates a real offline recommender system workflow with serving. First the data is obtained from a database, which is then processed and trains a Collaborative Filtering model. The model is then saved end served using Flask from a container.

## Database

The data I am using is the retailrocket e-commerce [dataset](https://www.kaggle.com/retailrocket/ecommerce-dataset) from kaggle. I did set up a docker-compose.yml file, which sets up the MySQL server I am going to use. Into the server I set up an "events" database and loaded the events.csv into an events table, out of which I am going to read the data using SQLAlchemy.

## Model training

This is a separate process to train the model. I am going to train a Neural Collaborative Filtering model (NCF), which I implemented from the [paper](https://arxiv.org/pdf/1708.05031.pdf). The model can be found in the ncf_model.py, which is used by the train.py. The train.py contains all the required steps for the model training - data preparation and model train. Note that for now there is no test set as my aim was not to optimise a model.

### Data preparation

SQLAlchemy is used to read in the data from the SQL server and then I call the parts of my train class. Note that this process has logging and has some minor logic implmeneted such as

- To each user we have to have at least 3 logs on any item to be considered in the dataset, furthermore I used the mean plus two standard deviation to filter out outliers
- Each user has to visit at least two unique items in the investigated time period
- I used a moving time window, which length is specified in the config file
- The scoring of user-item relations is pretty simple 1 if viewed, 2 if added to basket and 3 if purchased
- The mappings of user and item ids are written so that we can map them back and forth during serving also

### Model training

The training is done on GPU if available and the trained model is saved for serving.

## Model serving

The serving is containerized and the image can be build using the Dockerfile. The serving is in the main.py file, using simply flask and the url has to be '/topN=<top_n>for_user=<user_id>', where top_n is the number of top 10 products we want to offer and user_id is the user we want to show personalized offers.

Here further logical approaches could be applied, such as categorical or instock filtering.
