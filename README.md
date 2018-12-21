# CSPN implemented in Pytorch 0.4.1


### Introduction
This is a PyTorch(0.4.1) implementation of [Depth Estimation via Affinity Learned with Convolutional Spatial Propagation Network](http://arxiv.org/abs/1808.00150). At present, we can provide train script in NYU Depth V2 dataset for depth completion and monocular depth estimation. KITTI will be available soon!

Note: we fix some bugs in [original code](https://github.com/XinJCheng/CSPN/issues/9).

### Reselt
We use 2 Titan X to train CSPN for depth completion and monocular depth estimation.  

#### Monocular Depth Estimation

 Method |   rml  | rmse  | log10 | Delta1 | Delta2 | Delta3 
 :-------| :------: | :------: | :------: | :------: | :------: | :------: 
 CSPN_ours  | 0.151 | 0.546 | 0.064 | 0.793 | 0.949 | 0.985
 
![avatar](https://lh3.googleusercontent.com/QscRiaAp4vf8E8ieeW7I7JgDI8CAVtLJwDZPSZN3kGrgN5NPfPTTryqN8NzhlIfk_SjkYFvSCyLYcXVrRsJcyo9Je4b_JUU8Dmvx6gkOS6qUk7-kaZjklXmZJQY_sfcIYK6AXDJKNTXmpDIf05nyLmzuNQupdl_6obIoSPheJHw2HsEh97ODbjkbY02m13AGBZ_nrs6vkYTOn0dvtkrPoCDN4nUNBXUoh5mfp-CbMr0eGMSBlb4zp9mprG9yQPwVQvnz5xSopClEiMN0_QUcu-44hAtmIPo8p6-slmmZhdEu3Ub-50IYGxiG_taIjx0tljlw480b9a6xSrCjYBv8RnVBdA2GyfBW9vBbtvUrm-c-8xIV46Y5nhy30PdzvMAcwx2m7_2hGKjlGh3ynPYsXkVsHulUAOBwfHgbrCG4WSCENeXLQAqHwzqRyYMDSNkY-VY05KIXoWD0gIAFMj96gCqWiH3GTrGEcaKwSa80u9gBL5PEcfEqfzvyF_5RLugY6VncSqaWM0EKAEVq6RDqoLEDUWuK44EusSbgGRlg4LqnYGSVDTEMmcz_QvJg2vbz2vInEFQl0ty1OwDIpJkxDTk9sBmiIAYWJUY3B6y9EdXqQ23_aXSynctqJTSolae-GkEdwbpTTWoRns4V3e-r67vB=w633-h1264-no)
 
#### Depth Completion

We test CSPN for depth completion in NYU Depth dataset and use 500 sparse samples. 

 Method |   rml  | rmse  | log10 | Delta1 | Delta2 | Delta3 
 :-------| :------: | :------: | :------: | :------: | :------: | :------: 
 CSPN   | 0.016  | 0.117 | - | 0.992| 0.999 | 1.000
 CSPN_ours  | 0.023 | 0.152 | 0.010 | 0.988 | 0.997 | 0.999
 
 ![avatar](https://lh3.googleusercontent.com/-RHED87Kk-52yT1XzOSiwa6WQ4ixA8Cj0HpBOebxMNisPAsjtV6SPlpji4YikXy7v_Ahb5HSO_CX10NMAYf0ZfDoLztpFq7QtRH3gAhYuBrCA0l7w1p7YGGZUHr9dOGBUhnh8fPztZC1-Nmlod1ikekLVJZ0V-JC6e6gtfPYx9MSRzU_OvoM4EQu0kCOOAhsbcmmqd1mAVJHQp2s3FS0K6wRy_iQOpVZRXCknnspIWruMfmjXaDjiF-zPPjF0iANfLQzD3TlZwuVozUdmuuYlK7jUB6Je3AX1ueHUmy5t0Xx5kXSZrhfBD5tY2_Y06dTKdio40vEkj67BXpy8EvlYNlmIPwYRL-q21hXhLp4M3nYENjimHIeXjuHhWRYSsg0IVB8a9jGvyfsFe_0wAPVsswmwgGAdWNutnBJXchtWj6-9eHhTMHdZHZPd3r9PqkQvLkXPbb8YhdRRAImr3y2Xz6zcxFgNSyuKU94UjI0Ocb1Ugpu0NiTOUhoDibIUYhQ0MVHhjCF_ddaN2CHb2P-3IDmraqHBOfqw0GBEsiyUP_7sM2o6xRguDnNNzwaWezH0uF73EMwii5bMCBO-TQRK2iWMYiMklelrdECO8ph5e6nASaXjuHwDuNCcsYfYW9y0zEe19klkPNBqiJy7v_vOwU=w843-h1264-no)
