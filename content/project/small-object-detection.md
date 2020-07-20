---
title: Small Object Detection
date: '2020-07-20T18:40:06.000+00:00'
tags:
- Deep Learning
- " Object Detection"
- Small Object Detection
post: ''
frontpic: "/uploads/comparison-results-strides-1.png"
draft: true

---
!\[Detecting a small tennis ball with the baseline Mobilenet V1 and our Modified Backbones\](static/uploads/comparison-results-strides-1.png)

In this work, we present a simple solution for increasing small object detection performance. Through a series of empirical experiments we analyze the effect of aggressive down-scaling of feature maps in a convolutional backbone. While this strategy suits location invariant tasks such as image classification, it does not work well with location variant tasks such as object detection. This particularly affects performance of object detection for small objects as the aggressive down-scaling results in suppression of discriminative features belonging to small objects. We show that instead of up-sampling the input image, the key improvement comes from revisiting the down-scaling strategy. In particular, ablation studies on the mobilenet v1 backbone and SSD architecture show the most improvement from removing strides in the first layer of the backbone.