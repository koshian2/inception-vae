# Inception-vae
Variational Auto Encoder using Inception module in PyTorch

# Summary
Improving the blurring peculiar to VAE using the Inception module. 

## Inception VAE(Proposed method)
![](https://github.com/koshian2/inception-vae/blob/master/images/reconstruction_inception_01.png)

## CNN VAE
![](https://github.com/koshian2/inception-vae/blob/master/images/reconstruction_normal_05.png)

# Changing an inception module
Original inception is concatenation, but use simple addition instead.
![](https://github.com/koshian2/inception-vae/blob/master/images/inception_vae_model.png)

# Experiment
| Name        | Inception | # repeat | # params(M) | F/B pass size(MB) | batch_size |
|-------------|-----------|----------|------------:|------------------:|------------|
| Normal-1    | No        | 1        |      1.96 M |            227.07 | 32         |
| Normal-2    | No        | 2        |      3.53 M |            337.32 | 32         |
| Normal-3    | No        | 3        |      5.10 M |            447.57 | 32         |
| Normal-4    | No        | 4        |      6.67 M |            557.82 | 32         |
| Normal-5    | No        | 5        |      8.24 M |            668.07 | 32         |
| Normal-6    | No        | 6        |      9.81 M |            778.32 | 32         |
| Normal-8    | No        | 8        |     12.95 M |            998.82 | 16         |
| Normal-10   | No        | 10       |     16.09 M |           1219.32 | 16         |
| Inception-1 | Yes       | 1        |      5.95 M |            709.41 | 32         |
| Inception-2 | Yes       | 2        |     11.51 M |           1302.00 | 16         |

* Batch_size of Normal-08, Normal-10, and Inception-02 are halved due to GPU memory.
* F/B pass size and # params are calculated by [pytorch-summary](https://github.com/sksq96/pytorch-summary).

# Result
![](https://github.com/koshian2/inception-vae/blob/master/images/inception_vae_result.png)

# See details(Japanese)
https://qiita.com/koshian2/items/e2d05d9151f5ae9deefb

# Reference
This code was experimented with [The Japanese Female Facial Expression (JAFFE) Database](http://www.kasrl.org/jaffe.html). Thank you.

> Michael J. Lyons, Shigeru Akemastu, Miyuki Kamachi, Jiro Gyoba.  
Coding Facial Expressions with Gabor Wavelets, 3rd IEEE International Conference on Automatic Face and Gesture Recognition, pp. 200-205 (1998).

[Original Inception papaer](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Szegedy_Going_Deeper_With_2015_CVPR_paper.pdf):

> Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich.  
Going Deeper with Convolutions, CVPR2015
