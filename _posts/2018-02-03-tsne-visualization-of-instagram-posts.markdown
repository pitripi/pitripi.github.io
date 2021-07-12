---
layout: post
title: "t-SNE Visualization of Instagram Posts"
description: "In this post, I visualized instagram images in a square grid."
date: 2018-02-03 13:15:56 +0630
image: '/images/tsne/tsne_title.jpeg'
tags:   [tech, t-sne, visualization]
---

A year ago, I read [Andrej's blog](https://karpathy.github.io/2015/10/25/selfie/) post where he analyzed selfies using Convolutional Neural Networks. I was intrigued by the visualization technique he used which grouped images in such a way that nearby images were similar. In this post, I visualised my own collection of images in a square grid.

### Steps Involved
t-Distributed Stochastic Neighbor Embedding (t-SNE) is a technique for visualizing high dimensional data in 2D or 3D. It defines a cost function between a joint probability distribution,P, in the high-dimensional space and a joint probability distribution, Q, in the low-dimensional space and minimizes that cost function. In this case, we're plotting images but it can be used with all other kinds of data.

![tsne_mnist]({{site.baseurl}}/images/tsne/tsne_mnist.png)
*t-SNE visualization of MNIST data.*

![tsne_jpeg]({{site.baseurl}}/images/tsne/tsne_mnist_grid.jpeg)
*t-SNE visualization of MNIST data in a grid. (Not related to the previous visualization).*

The full script can be found [here](https://github.com/pitripi/tsne-grid). The script requires keras and tensorflow. It is tested with tensorflow (1.4.0) and keras (2.1.1).

**Build model**: VGG16 network (without top fc layers) is used and outputs of last conv block are flattened to form a vector.

**Load images**: All the images from the source directory are loaded.

**Generate high dimensional representations**: A single forward pass through the network generates the high dimensional representations.

**Get 2D point locations**: t-SNE implementation of scikit-learn converts these representations to 2D data points.

**Distribute 2D representations in a square grid and save images**: Finally, jonker-volgenant algorithm distributes these 2D points into a square grid and we assign every point in the grid a small image.


![tsne_rand]({{site.baseurl}}/images/tsne/tsne_rand.jpeg)
*t-SNE visualization of some random images in a grid.*

### System Details
HP Pavilion

4 GB RAM

Nvidia GeForce GT 740M

### __References__
1. L.J.P. van der Maaten and G.E. Hinton. Visualizing High-Dimensional Data Using t-SNE. Journal of Machine Learning Research 9(Nov):2579–2605, 2008.[\[PDF\]](https://lvdmaaten.github.io/publications/papers/JMLR_2008.pdf)[\[Supplemental material\]](https://lvdmaaten.github.io/publications/misc/Supplement_JMLR_2008.pdf)[\[Talk\]](https://www.youtube.com/watch?v=RJVL80Gg3lA&list=UUtXKDgv1AVoG88PLl8nGXmw)[\[Code\]](https://lvdmaaten.github.io/tsne/)