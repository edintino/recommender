# import config
import sys
sys.path.append("/home/eugenio/PycharmProjects/recommender_system")
sys.path.append("../")

from config import *

import pandas as pd

import pickle5 as pickle

import torch

from ncf_model import NCF

import flask


def prepare_serving():
    with open('user_mapper.pkl', 'rb') as file:
        user_mapper = pickle.load(file)
    with open('item_mapper.pkl', 'rb') as file:
        item_mapper = pickle.load(file)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = NCF(len(user_mapper), len(item_mapper), **NCF_PARAMS).to(device)
    model.load_state_dict(torch.load('model.pt', map_location=torch.device('cpu')))
    model.eval()

    return user_mapper, item_mapper, device, model


def recommend_for_user(user_id, items, model, device):
    users = torch.Tensor([user_id]).long()
    items = torch.Tensor(items).long()

    cross_prod = torch.cartesian_prod(users, items)

    prediction_dataset = torch.utils.data.TensorDataset(cross_prod[:, 0], cross_prod[:, 1])
    prediction_loader = torch.utils.data.DataLoader(dataset=prediction_dataset,
                                                    batch_size=4096,
                                                    shuffle=False)

    results = pd.DataFrame(columns=['user', 'item', 'predictions'])

    for user, item in prediction_loader:
        with torch.no_grad():
            # logger.debug(f"start prediction")
            predictions = model(user.to(device),
                                item.to(device)).detach().cpu().numpy().reshape(-1)
            # logger.debug("prediction ok")
            tmp_pd = pd.DataFrame({'user': user.detach().cpu().numpy().reshape(-1),
                                   'item': item.detach().cpu().numpy().reshape(-1),
                                   'predictions': predictions})

            results = pd.concat([results, tmp_pd])

    return results.sort_values('predictions', ascending=False).reset_index(drop=True)

app = flask.Flask(__name__)


@app.route('/topN=<top_n>for_user=<user_id>')
def recommendation(top_n, user_id):
    # load
    user_mapper, item_mapper, device, model = prepare_serving()

    # user_id = list(user_mapper[user_mapper.keys())[0]])
    df = recommend_for_user(int(user_id), range(len(item_mapper)), model, device)

    return df[:int(top_n)].to_html(header="true", table_id="table")


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
