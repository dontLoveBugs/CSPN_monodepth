# CSPN implemented in Pytorch 0.4.1


### Introduction
This is a PyTorch(0.4.1) implementation of [Depth Estimation via Affinity Learned with Convolutional Spatial Propagation Network](http://arxiv.org/abs/1808.00150). At present, we can provide train script in NYU Depth V2 dataset for depth completion and monocular depth estimation. KITTI will be available soon!

Note: we fix some bugs in [original code](https://github.com/XinJCheng/CSPN/issues/9).

### Reselt
We test CSPN for depth completion in NYU Depth dataset and use 500 sparse samples. 

 Method |   rml  | rmse  | log10 | Delta1 | Delta2 | Delta3 
 :-------| :------: | :------: | :------: | :------: | :------: | :------: 
 CSPN   | 0.016  | 0.117 | - | 0.992| 0.999 | 1.000
 CSPN_ours  | 0.023 | 0.152 | 0.010 | 0.988 | 0.997 | 0.999
 
 ![avatar](https://lh3.googleusercontent.com/jw26V6bAVDI1zM0f2R2aEyAGhgApQj9C98QqbxcyoJ2BCVsF5dn1s5g7MFmrqE8b8gW_LmPAj313FrW2C3aladLfJHHyZStvth1RPfZiIPduIgkUjHF8544i85WJD6QPiB9VA8CkrrZZAm0Isd8jjMFl8I5jCmUZiMNOog9wm_8Bxa_AcOQYxKQTjRXrTWVJMNHEY8TXK_OMmNc19B_mUQ_lvF1-2TA8nIWuRwLWDPLd6pwR5dXXZ7uIixLXZHbJdDYgPQnVJkPwXTd9i2qE_OGXcTVu3nIT88kNki1oyyYXqFLY6C6BrVjxBHI5u_wfBln08IoIdUJ_KfCwKzNCgL7hbbMGivowf_U5Te9AOC6gdtSxwnaDTQQmSv777iA_OrO7hoi-AlRwJVkB2f2sIcBil3ttNL969lvKOKn9mpGsvJqy2LsqXF1oEUVwl3aeIHASipFAU1gVbxGD7uQ0nl_BKjtJYIpm_cF4p3iVBdYajEdUFu2EuP1iJa4CEW-av1EPW1w9A73ME6RDw7V_DrvamUnxWyY-5tVNl8U_CV3d0amqM3gRsrelz_y7unNr4EG6vrXvVLvHA1uSf6TL5msa9WXk4k-PzLyeaHbBva410Hg_E2K3PsllHf1lnwWJAP94DzDG68ZYr8vgk6qpJFc=w843-h1264-no)
