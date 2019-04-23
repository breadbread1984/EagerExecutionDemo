#!/bin/bash

# install tensorflow repo
echo "deb [arch=amd64] http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | sudo tee /etc/apt/sources.list.d/tensorflow-serving.list && \
curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | sudo apt-key add -
# install tensorflow serving tools
sudo apt update
sudo apt install tensorflow-model-server
pip3 install requests
# generate serving model and test it
python3 convert_model.py
saved_model_cli show --dir ./mnist/1 --all # checkout output model
saved_model_cli run --dir ./mnist/1 --tag_set serve --signature_def serving_default --input_exp 'input_1=np.random.normal(size=(1,28,28,1))' # test serving
# start serving at background
cp -rv ./mnist /tmp # serving model needs an absolute path
nohup tensorflow_model_server --rest_api_port=8501 --model_name=mnist --model_base_path="/tmp/mnist" >server.log 2>&1 &
# send test request to the server
python3 send_request.py

