# EagerExecutionDemo
The project gives an example on how to use eager execution to train mnist model. Eager execution is a new mode provided by Tensorflow for machine learning researchers. The mode greatly facilitates network debugging and gradient manipulation. Hope the example can help tensorflow developers with their projects.

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
