+++
draft = true
frontpic = ""
post = ""
tags = []
title = "Power of Hooks in Pytorch"

+++
## What are hooks?

Pytorch allows you to custom function calls to its **module** and **tensor** objects. The calls can both be added to the forward method of the object as well as the backward method. A hook added to the forward method will be called with the following arguments

1. The instance of the module itself
2. The input to the module
3. The output of the forward method

## Why hooks?

Now that we know what hooks are, it's important to understand their use case. Most commonly, they are either used for debugging purposes, calculating model size or calculating the number of ops. Let's say you imported a backbone from the **torchvision** package such as resnet18. If you wanted to calculate the number of ops for each layer you might be tempted to rewrite the entire resnet18 backbone with commands added to the forward method for calculating the ops. Instead a better way is to add a hook to the module without re-writing the code for resnet18.

## Example: Adding Dropout to a Module

Let's demonstrate the power of hooks with an example of adding dropout after every conv2d layer of a CNN. Some knowledge of Python decorators is required to understand the code but other than it is quite simple. 

For this example, let's assume a ResNet18 backbone for image classification. We want to add dropout after every convolutional layer. 

### Adding the Decorator

### Testing it Out

## References

* [https://python-3-patterns-idioms-test.readthedocs.io/en/latest/PythonDecorators.html](https://python-3-patterns-idioms-test.readthedocs.io/en/latest/PythonDecorators.html) "Decorators in Python3")
* [http://funkyworklehead.blogspot.com/2008/12/how-to-decorate-init-or-another-python.html](http://funkyworklehead.blogspot.com/2008/12/how-to-decorate-init-or-another-python.html "Passing an object instance through a decorator")