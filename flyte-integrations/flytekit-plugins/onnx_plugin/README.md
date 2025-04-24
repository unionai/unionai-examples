# ONNX

Open Neural Network Exchange ([ONNX](https://github.com/onnx/onnx)) is an open standard format for representing machine learning
and deep learning models. It enables interoperability between different frameworks and streamlines the path from research to production.

The flytekit onnx type plugin comes in three flavors:


## ScikitLearn

```
$ pip install flytekitplugins-onnxpytorch
```

This plugin enables the conversion from scikitlearn models to ONNX models.


## TensorFlow

```
$ pip install flytekitplugins-onnxtensorflow
```

This plugin enables the conversion from tensorflow models to ONNX models.


## PyTorch

```
$ pip install flytekitplugins-onnxpytorch
```

This plugin enables the conversion from pytorch models to ONNX models.

## Other frameworks

If you'd like to add support for a new framework, please create an issue and submit a pull request to the flytekit repo.
You can find the ONNX plugin source code [here](https://github.com/flyteorg/flytekit/tree/master/plugins).
