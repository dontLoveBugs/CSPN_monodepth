# CSPN implemented in Pytorch 0.4.1


### Introduction
This is a PyTorch(0.4.1) implementation of [Depth Estimation via Affinity Learned with Convolutional Spatial Propagation Network](http://arxiv.org/abs/1808.00150). At present, we can provide train script in NYU Depth V2 dataset for depth completion and monocular depth estimation. KITTI will be available soon!

Note: we fix some bugs in [original code](https://github.com/XinJCheng/CSPN/issues/9).

### Reselt
We test CSPN for depth completion in NYU Depth dataset. We use 500 sparse samples.

 Method |   rml  | rmse  | log10 | Delta1 | Delta2 | Delta3 
 :-------| :------: | :------: | :------: | :------: | :------: | :------: 
 CSPN   | 0.016  | 0.117 | - | 0.992| 0.999 | 1.000
 CSPN_ours  | 0.023 | 0.152 | 0.010 | 0.988 | 0.997 | 0.999
 
 ![avatar](https://lh3.googleusercontent.com/P0M0bgTVrbQduZDkZisVJYe9rIjo0avKR530T1yfLnBRCNHCSOEUXcmpAA5SoShwhX5GFooLb0geFLLZTyp99HbQz0UoFma8dFrC-mm3QUMPBG41PD1VxhRIDbz5Z8W1febE7f_oIJ4HXEDv_W6D-vD1B1cmmQMD9ZUSw5AeZYAHJnoFa7OJf3VvnXaBLYZM3TAlSbAOSfuGaqF9tTPMJYBG6cCPIojQh88t9XhQ0pGDBvXpbPJliChz1DEYiyp1rYbSa82sKbmdb84Ap3FqQBqb3yHCwXBeA1UXQIL0cS64ycf-fuJJ-HZkPyRAV29R0j89kngw6cJaFyzyQ-WPwaSwb4CF3AA_9fv7dmWArsS2vqby75tI-Z6T5zUnF3jnEPMHWSstZjSePfQSF73LLF-3sali9klAKltVFRe1nlIYaq2OyMuAYx_KKWoa-nqqrgtAy6PoBriR8B-mouYiD-Nai6z5GQ3wiuJWnbuWD4hoEW-kZMnl8D1VEHSnlgdLKaQLDUug0zkPEnMEKESBSKqyadVO9gDzeNsH3-Wq5en4ulmvWGGebv2BWXkmBlRAkSd7KvvASk8uyLx2hX2z5jphhPx3ZF5AU9ocLRx-rNXQAWj72LdW6c0-xttIZizHBkIDpMmrhKG0BStToXvpg5A=w843-h1264-no)
