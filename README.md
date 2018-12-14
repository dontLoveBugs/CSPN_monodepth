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
 
 ![avatar](https://lh3.googleusercontent.com/lA1zTX-yZPJWDW9F4LfR9vplXpJkFTW6wFPyZ0Y-QrDZORrh64kbfJ1U9yUsmQA22ALn0JoaTzTuFokwc7cJjqa2nUx2kMgqiT0yTs9yU-AS_Kv2jvuW8ZkK6FnlAIlKrRUgdPr4VIpmp35X0Zpu4uieolRGbOlfnECbSXiTPrwnCmDQejKR7WIcKa6YkTdkMLIxAOBN27HWbcfFnhQaTcGENmlSSSZS0g6o0N8trc_1QTKdFstT2kYOI_lWrHBSSeB87Omj3z7TmRa_c9TC30euua0NI2dHA58qeQoNs9Tf1yCadO7lW493oz1IsdHwVIl0ecybX2IikxZt0d8Y14QlxTJaY6RHZrazWe5H0nzdKIwa-KhwrQd38Eilq76rPRJGaLG7kcgWD_b0NxKSS7AiyErOhyxDlRmHWQRfCvyqTrt-_qK-5D-WWqgouW81IaYSOvhuKgZwsSqdNHcan5tb-DTMyM4A9vE-_P1GJNM-uePsdYnNLP4vPTKzCmxrKg47c6LCrKD-1asuPV2fxvzOPH5qsVApyIeslVVC_bTZbYn9tmKbDoDlETZjAAJp5vpPjZ-5oijJqoACIbLw-XU1N6d5mkxKp_T6SaRDYvIAraSct-Uz3a6XzHRoqSI5rgw_K38zLqfFmyDSS-gGYaHx=w602-h1203-no)
 
#### Depth Completion

We test CSPN for depth completion in NYU Depth dataset and use 500 sparse samples. 

 Method |   rml  | rmse  | log10 | Delta1 | Delta2 | Delta3 
 :-------| :------: | :------: | :------: | :------: | :------: | :------: 
 CSPN   | 0.016  | 0.117 | - | 0.992| 0.999 | 1.000
 CSPN_ours  | 0.023 | 0.152 | 0.010 | 0.988 | 0.997 | 0.999
 
 ![avatar](https://lh3.googleusercontent.com/PaRCgrxuWrBPnlaoRvYFa1QrMhiV6JqYIBAYFbAnPVIrkqC9YTOVTcYP6ZZnrn6uZXjZIGRqeBufcXHwIqrsFkyjCtKJ6FtMxTXy7dawSWtQKTgYXaLNzY77iIV8DZreAAUWtfdZkSm6wTsfGZWAF5cQO3CQQ_7ZtXJRWZ7WGEp0hNqUeeR6DOjL2vei_sr-d1Ir2K_LAL6rvX8LpaM8M8b1RAAhOMgTozeE7RRqU7Eq-G3fhD5cbFMAuysLv5W21VnVvvqoFk7pgDyelrB0DB0zp5LoLkwte6cA1kupVF4BQDf56P1pdEpkG2lovV1cIOyTv6XyaVMRdYNVHc8R-xVxc3f44IUppLcQwzeaaNIFWTaZ5adR_6lJGSmffMZsRNTQhEPt9Yp3907BhukXsnWG7jb23zan4l2IaA78kIGY2rd2bvpX9yuvOFVpnmEiHqUGVngmaSB1DGj9SmGsSZOxsT402RHZsUkICsmCY73RgRKfstfGxOTr5ZuHHRtaKKLPLyMDkTi349bMeJ-lm159Qu_WgT9jActwROp1XU3TmL1wq9TrJinVIPtuSmebDUix_Y2AyT75IDmiq7sZ0nLpzdUexyHtAEhLdPEP8rxJL0lGXit8F0LFdbNqn_NS1laEZ9afr0t92dLrb0-uEqU=w803-h1203-no)
