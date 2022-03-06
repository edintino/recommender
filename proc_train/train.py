# import config
from config import *

import logging
logging.basicConfig(filename='train.log',
                    encoding='utf-8',
                    filemode='w',
                    level=logging.DEBUG)
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
logging.getLogger('').addHandler(console)


import sys
import pickle
import pandas as pd
from datetime import timedelta

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

sys.path.append("../")
from ncf_model import NCF

class NCFTrain:
    def __init__(self):
        self.events = None
        self.user_mapper = None
        self.item_mapper = None

        self.N = None
        self.M = None

        self.train_loader = None

    def prepare_data(self, events):
        """Prepares the data for the model training and prepares
        required files for the serving of customers also."""
        self.events = events

        logging.info(f"Events row count: {len(self.events)}")
        logging.info(f"Unique visitor count: {self.events['visitorid'].nunique()}")
        logging.info(f"Unique item count: {self.events['itemid'].nunique()}")

        counts = self.events.groupby('visitorid')['itemid'].count()
        unique_items = self.events.groupby('visitorid')['itemid'].nunique()

        self.events = self.events[self.events['visitorid'].isin(
            counts.index[counts.between(3, counts.mean() + 2 * counts.std())])]
        logging.info(f"Row count after outlier filtering: {len(self.events)}")

        self.events = self.events[self.events['visitorid'].isin((unique_items.index[unique_items > 1]))]
        logging.info(f"Row count after multiple item filtering: {len(self.events)}")

        self.events['timestamp'] = pd.to_datetime(self.events['timestamp'], unit='ms')
        self.events = self.events[self.events['timestamp'] > (self.events['timestamp'].max() - \
                                                              timedelta(TRAIN_TIME_WINDOW))]
        logging.info(f"Row count after time window filtering: {len(self.events)}")

        self.events = self.events[['visitorid', 'itemid', 'event']]
        self.events['event'] = self.events['event'].map(EVENT_SCORES)

        self.events = self.events.groupby(['visitorid', 'itemid'])['event'].max().reset_index()
        self.events.drop_duplicates(inplace=True)

        logging.info(f"Final row count (max event): {len(self.events)}")

        self.user_mapper = dict(zip(self.events['visitorid'].unique().astype(int), range(len(self.events['visitorid'].unique()))))
        self.item_mapper = dict(zip(self.events['itemid'].unique().astype(int), range(len(self.events['itemid'].unique()))))

        logging.info(f"Writing user mapper")
        with open("user_mapper.pkl", "wb") as file:
            pickle.dump(self.user_mapper, file, protocol=pickle.HIGHEST_PROTOCOL)

        logging.info(f"Writing item mapper")
        with open("item_mapper.pkl", "wb") as file:
            pickle.dump(self.item_mapper, file, protocol=pickle.HIGHEST_PROTOCOL)

        self.events['new_visitorid'] = self.events['visitorid'].map(self.user_mapper)
        self.events['new_itemid'] = self.events['itemid'].map(self.item_mapper)

        self.N = len(self.events['new_visitorid'].unique())
        self.M = len(self.events['new_itemid'].unique())

        user_ids = torch.from_numpy(self.events['new_visitorid'].values).long()
        item_ids = torch.from_numpy(self.events['new_itemid'].values).long()
        preference = torch.from_numpy(self.events['event'].values)

        train_dataset = TensorDataset(
            user_ids,
            item_ids,
            preference)

        self.train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                        batch_size=TRAIN_BS,
                                                        shuffle=True)

    def train_model(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logging.debug(f"The model us trained on {device}")

        model = NCF(self.N, self.M, **NCF_PARAMS)
        model.to(device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adagrad(model.parameters())

        model.fit(self.train_loader, criterion, optimizer, device, EPOHCS)

        torch.save(model.state_dict(), 'model.pt')
