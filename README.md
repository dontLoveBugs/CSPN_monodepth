# CSPN implemented in Pytorch 0.4.1


### Introduction
This is a PyTorch(0.4.1) implementation of [Depth Estimation via Affinity Learned with Convolutional Spatial Propagation Network](http://arxiv.org/abs/1808.00150). At present, we can provide train script in NYU Depth V2 dataset for depth completion and monocular depth estimation. KITTI will be available soon!

### Faster Implementation
We re-implement CSPN using [Pixel-Adaptive Convolution](http://arxiv.org/abs/1904.05373). 

### Multi_GPU
The implementation of multi-gpus is based on [inplace abn](http://arxiv.org/abs/1712.02616).

### Results
TODO

### Third Libs
[inplace abn](https://github.com/mapillary/inplace_abn)
[Pixel-Adaptive Convolution](https://github.com/NVlabs/pacnet)
