# EagerExecutionDemo
The project gives an example on how to use eager execution to train mnist model. Eager execution is a new mode provided by Tensorflow for machine learning researchers. The mode greatly facilitates network debugging and gradient manipulation. Hope the example can help tensorflow developers with their projects.

## How to plot network structure
in order to show the structure of the network, you need to install some packages with the following command

```bash
sudo apt install python3-pydot python3-pygraphviz
```

plot the network structure with

```bash
python3 MNISTModel.py
display model.png
```

## How to generate dataset
generate mnist dataset in tfrecord format with the following command

```Bash
python3 create_dataset.py
```

## How to train model
train LeNet with the following command

```Bash
python3 train_mnist.py
```

## How to test model
test LeNet with the following command

```Base
python3 test_mnist.py
```

## How to predict with trained model
an example is given in predict.py. you can run it with

```Base
python3 MNISTModel.py <grayscale img of size 28x28>
```

the predicted class of the input image will be printed on the console.

## How to monitor the training with tensorboard
monitor the training process with command

```Bash
tensorboard --logdir model
```

## How to serve the model
executing the command

```Bash
bash start_serving.sh
```

## How to test the server
executing the command

```Base
python3 send_request.py
```

to send input tensor data to URI, and get return tensor.

