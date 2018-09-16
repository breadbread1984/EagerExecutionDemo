# EagerExecutionDemo
The project gives an example on how to use eager execution to train mnist model. Eager execution is a new mode provided by Tensorflow for machine learning researchers. The mode greatly facilitate network debugging and gradient manipulation. Hope the example can help tensorflow developers with their projects.

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

## monitor the training with tensorboard
monitor the training process with command

```Bash
tensorboard --logdir model
```
