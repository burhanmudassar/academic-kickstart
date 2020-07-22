+++
draft = true
frontpic = ""
post = ""
tags = []
title = "Power of Hooks in Pytorch"

+++
## What are hooks?

Pytorch allows you to add custom function calls to its **module** and **tensor** objects called **hooks**. The calls can both be added to the forward method of the object as well as the backward method. A hook added to the forward method will be called with the following arguments

1. The instance of the module itself
2. The input to the module
3. The output of the forward method

```
    def dropout_hook(self, module, input, output):
    	output = F.dropout2d(output, self.prob, True, False)
        return output
```

## Why hooks?

Now that we know what hooks are, it's important to understand their use case. Most commonly, they are either used for debugging purposes, calculating model size or calculating the number of ops. Let's say you imported a backbone from the **torchvision** package such as vgg16. If you wanted to calculate the number of ops for each layer you might be tempted to rewrite the entire vgg16 backbone with commands added to the forward method for calculating the ops. Instead a better way is to add a hook to the module without re-writing the code for vgg16.

## Example: Adding Dropout to a CNN

Let's demonstrate the power of hooks with an example of adding dropout after every conv2d layer of a CNN. For example, This is the code for the ResNet module in the torchvision package. Needleesly to say adding dropout to the output of each layer is not as trivial.

    class ResNet(nn.Module):
    
        def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                     groups=1, width_per_group=64, replace_stride_with_dilation=None,
                     norm_layer=None):
            super(ResNet, self).__init__()
            if norm_layer is None:
                norm_layer = nn.BatchNorm2d
            self._norm_layer = norm_layer
    
            self.inplanes = 64
            self.dilation = 1
            if replace_stride_with_dilation is None:
                # each element in the tuple indicates if we should replace
                # the 2x2 stride with a dilated convolution instead
                replace_stride_with_dilation = [False, False, False]
            if len(replace_stride_with_dilation) != 3:
                raise ValueError("replace_stride_with_dilation should be None "
                                 "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
            self.groups = groups
            self.base_width = width_per_group
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                                   bias=False)
            self.bn1 = norm_layer(self.inplanes)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.layer1 = self._make_layer(block, 64, layers[0])
            self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                           dilate=replace_stride_with_dilation[0])
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                           dilate=replace_stride_with_dilation[1])
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                           dilate=replace_stride_with_dilation[2])
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512 * block.expansion, num_classes)
    
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
    
            # Zero-initialize the last BN in each residual branch,
            # so that the residual branch starts with zeros, and each residual block behaves like an identity.
            # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
            if zero_init_residual:
                for m in self.modules():
                    if isinstance(m, Bottleneck):
                        nn.init.constant_(m.bn3.weight, 0)
                    elif isinstance(m, BasicBlock):
                        nn.init.constant_(m.bn2.weight, 0)
    
        def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
            norm_layer = self._norm_layer
            downsample = None
            previous_dilation = self.dilation
            if dilate:
                self.dilation *= stride
                stride = 1
            if stride != 1 or self.inplanes != planes * block.expansion:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    norm_layer(planes * block.expansion),
                )
    
            layers = []
            layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                                self.base_width, previous_dilation, norm_layer))
            self.inplanes = planes * block.expansion
            for _ in range(1, blocks):
                layers.append(block(self.inplanes, planes, groups=self.groups,
                                    base_width=self.base_width, dilation=self.dilation,
                                    norm_layer=norm_layer))
    
            return nn.Sequential(*layers)
    
        def _forward_impl(self, x):
            # See note [TorchScript super()]
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
    
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
    
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
    
            return x
    
        def forward(self, x):
            return self._forward_impl(x)

### Adding the Hook

Let's write the hook that will do apply the dropout. The hook takes in 3 arguments i.e. the module itself, the input to the module and the output generated by forward method of the module. Our hook will just apply the dropout function to the output and overwrite it. The dropout2d arguments include the tensor to modify, the probability of dropping. The training flag can be set to be true only when the model is training or a custom combination of your choosing. Finally, inplace will overwrite the contents of output without creating a new tensor. This will raise an exception during training as autograd requires all outputs to be in memory for gradient propagation so we keep it as False.

        def dropout_hook(self, module, input, output):
            output = F.dropout2d(output, self.prob, True, False)
            return output

Now that the hook is ready, we need to register it to the model itself. The way to do that is to call the `register_forward_hook` method of the module with the handle of the `dropout_hook`. This will add the dropout hook to every layer of the model. We only need to add this to the output of the convolutional layers. For that we create another function called register_hook

    def register_hook(self, module):
    	if isinstance(module, nn.Conv2d):
        	module.register_forward_hook(dropout_hook)
    
    model.apply(register_hook)

The apply method is applied recursively to every nn.Module within the model so it is ensured that every conv2d layer will have the dropout hook added to it. For reconfigurability, let us create a dropout hook class that allows us to store the probability of dropping activation values as well. We will add a remove method as well that will remove the hooks added to the model.

    class DropoutHook():
    
        def __init__(self, prob):
            self.prob = prob
            self.handles = []
    
        def register_hook(self, module):
            if isinstance(module, nn.Conv2d):
                self.handles += [module.register_forward_hook(self.dropout_hook)]
    
        def dropout_hook(self, module, input, output):
            output = F.dropout2d(output, self.prob, True, False)
            return output
    
        def remove(self):
            for handle_ in self.handles:
                handle_.remove()

### Testing it Out

To test it out let's import the vgg16 backbone from the torchvision package. We will apply a random input to the model and store it for reference. We will then apply our dropout hook and evaluate the model in both training and evaluation mode and compare the outputs. 

    	import torch
        import torchvision
        model = torchvision.models.vgg16(pretrained=True)
    
        model.eval()
           
        x = torch.randn((1,3,224,224))
        refOut = model(x)
    
        dropout_ = DropoutHook(prob=0.2)
        model.apply(dropout_.register_hook)
    
        outWithDropout = model(x)
        
        dropout_.remove()
        outWithOutDropout = model(x)
    
        errDropoutModel = (outWithDropout - refOut).mean()
        errWithoutDropoutModel = (outWithOutDropout - refOut).mean()
        print("Dropout Model Error: {}, Non-dropout Model Error: {}".format(errDropoutModel, errWithoutDropoutModel))

The output clearly shows that the dropout hook is changing the outputs of the conv2d layers.

![](/uploads/dropout_result.PNG)

## Downsides

1. Hooks are not serializable which means so you cannot call torch.save on a model that has hooks
2. Hook references are not maintained in the model. Instead you have to store the handle to each hook (see the `DropoutHook` class for an example)

## References

1. [https://python-3-patterns-idioms-test.readthedocs.io/en/latest/PythonDecorators.html](https://python-3-patterns-idioms-test.readthedocs.io/en/latest/PythonDecorators.html) "Decorators in Python3")
2. [http://funkyworklehead.blogspot.com/2008/12/how-to-decorate-init-or-another-python.html](http://funkyworklehead.blogspot.com/2008/12/how-to-decorate-init-or-another-python.html "Passing an object instance through a decorator")