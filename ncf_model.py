import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd
from datetime import datetime

import logging

class NCF(nn.Module):
    def __init__(self, n_users, n_items, embed_dim, mlp_hidden,
                 gmf_hidden, drop_rate):

        super(NCF, self).__init__()
        self.N = n_users
        self.M = n_items
        self.D = embed_dim
        self.drop_rate = drop_rate

        self.u_mlp_emb = nn.Embedding(self.N, self.D)
        self.m_mlp_emb = nn.Embedding(self.M, self.D)
        self.u_gmf_emb = nn.Embedding(self.N, self.D)
        self.m_gmf_emb = nn.Embedding(self.M, self.D)

        # forward calculation from MLP branch
        self.mlp = nn.ModuleList([nn.Linear(2 * self.D, mlp_hidden[0])] + \
                                 [nn.Linear(mlp_hidden[i], mlp_hidden[i + 1]) for i in range(len(mlp_hidden) - 1)])

        # forward calculation from GMF branch
        self.gmf_fc = nn.Linear(self.D, gmf_hidden)
        # final calculation for NeuMF
        self.neuMF = nn.Linear(gmf_hidden + mlp_hidden[-1], 1)

        self.train_losses = []

    def forward(self, u, m):
        # output is (num_samples, D)
        u_mlp = self.u_mlp_emb(u)
        m_mlp = self.m_mlp_emb(m)
        u_gmf = self.u_gmf_emb(u)
        m_gmf = self.m_gmf_emb(m)

        # MLP ANN
        mlp_out = torch.cat((u_mlp, m_mlp), 1)

        for mlp in self.mlp[:-1]:
            mlp_out = F.dropout(F.relu(mlp(mlp_out)),
                                p=self.drop_rate,
                                training=self.training)
        mlp_out = self.mlp[-1](mlp_out)

        # GMF ANN
        gmf_out = u_gmf * m_gmf
        gmf_out = torch.sigmoid(self.gmf_fc(gmf_out))

        # NeuMF ANN
        return self.neuMF(torch.cat((mlp_out, gmf_out), 1))

    def fit(self, train_loader, criterion, optimizer, device, epochs):

        for it in range(epochs):
            t0 = datetime.now()

            train_loss = []
            for users, items, targets in train_loader:
                targets = targets.view(-1, 1).float()

                # move data to GPU
                users, items, targets = users.to(device), items.to(device), targets.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = self(users, items)
                loss = criterion(outputs, targets)

                # Backward and optimize
                loss.backward()
                optimizer.step()

                train_loss.append(loss.item())

            # Save losses
            self.train_losses.append(np.mean(train_loss))

            logging.debug(f'Epoch {it + 1}/{epochs}, Train Loss: {self.train_losses[-1]:.4f}, Duration: {datetime.now() - t0}')