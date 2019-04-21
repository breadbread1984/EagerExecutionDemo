#!/bin/bash

echo "deb [arch=amd64] http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | sudo tee /etc/apt/sources.list.d/tensorflow-serving.list && \
curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | sudo apt-key add -

sudo apt update
sudo apt install tensorflow-model-server

python3 convert_model.py
saved_model_cli show --dir ./serving_model --all # checkout output model
saved_model_cli run --dir ./serving_model --tag_set serve --signature_def serving_default --input_exp 'input_image=np.random.normal(size=(1,28,28,1))' # test serving

