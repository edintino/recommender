#!/bin/bash

docker-compose -f ./mysql/docker-compose.yml up -d

source /home/eugenio/anaconda3/bin/activate pytorch

python3 ./proc_train/main.py

cp ./config.py ./proc_serve/utils/
cp ./ncf_model.py ./proc_serve/utils/

mv -t ./proc_serve/utils/ item_mapper.pkl user_mapper.pkl model.pt

docker build ./proc_serve/. -t rec_serve

docker run rec_serve
