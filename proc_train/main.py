import sys
sys.path.append("/home/eugenio/PycharmProjects/recommender_system")
from config import DB_CONFIG

import sqlalchemy as db
from sqlalchemy.sql import text

import pandas as pd

from train import NCFTrain

def read_data():
    """Connect to MYSQL database and read in required data. """
    db_user = DB_CONFIG.get('user')
    db_pwd = DB_CONFIG.get('password')
    db_host = DB_CONFIG.get('host')
    db_port = DB_CONFIG.get('port')
    db_name = DB_CONFIG.get('database')
    # specify connection string
    connection_str = f'mysql+pymysql://{db_user}:{db_pwd}@{db_host}:{db_port}/{db_name}'
    # connect to database
    engine = db.create_engine(connection_str, pool_timeout=20, pool_recycle=299)
    connection = engine.connect()

    query = text('''

    select * from events;

    ''')

    return pd.DataFrame(connection.execute(query).fetchall(),
                        columns=['timestamp', 'visitorid', 'event', 'itemid', 'transactionid'])

if __name__ == '__main__':

    events = read_data()

    train_process = NCFTrain()
    train_process.prepare_data(events)
    train_process.train_model()

