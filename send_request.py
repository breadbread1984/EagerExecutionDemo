#!/usr/bin/python3

import json;
import requests;
import numpy as np;

x = np.random.normal(size=(1,28,28,1));
data = json.dumps({"signature_name": "serving_default", "instances": [{"input_1": x.tolist()}]});
headers = {"content-type": "application/json"};
json_response = requests.post('http://localhost:8501/v1/models/mnist:predict', data = data, headers = headers);
predictions = np.array(json.loads(json_response.text)["predictions"]);
print(predictions);

