FROM pytorch/pytorch:latest

RUN apt-get update \
     && apt-get install -y \
        libgl1-mesa-glx \
        libx11-xcb1 \
     && apt-get clean all \
     && rm -r /var/lib/apt/lists/*

COPY main.py utils/config.py utils/ncf_model.py utils/model.pt requirements.txt utils/item_mapper.pkl utils/user_mapper.pkl  ./

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

EXPOSE 5000

CMD [ "python3", "./main.py"]