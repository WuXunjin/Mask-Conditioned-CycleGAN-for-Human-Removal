# dynamic-object-removal-with-unpaired-images

# Description

GAN-based Application to remove objects (humans) in images and to reconstruct the occluded areas with plausible background.

keyword: dynamic object removal, generative adversarial network, CycleGAN, unpaired images

# Demo

1. Taking a single image with a human as an input

<div align="center">
<img src="https://github.com/hiroyasuakada/dynamic-object-removal-with-unpaired-images/blob/master/demo/input_1.jpg" alt="属性" title="タイトル">
<img src="https://github.com/hiroyasuakada/dynamic-object-removal-with-unpaired-images/blob/master/demo/input_2.jpg" alt="属性" title="タイトル">
<img src="https://github.com/hiroyasuakada/dynamic-object-removal-with-unpaired-images/blob/master/demo/input_3.jpg" alt="属性" title="タイトル">
</div>

<br>

2. Generating an image without the human

<div align="center">
<img src="https://github.com/hiroyasuakada/dynamic-object-removal-with-unpaired-images/blob/master/demo/output_1.jpg" alt="属性" title="タイトル">
<img src="https://github.com/hiroyasuakada/dynamic-object-removal-with-unpaired-images/blob/master/demo/output_2.jpg" alt="属性" title="タイトル">
<img src="https://github.com/hiroyasuakada/dynamic-object-removal-with-unpaired-images/blob/master/demo/output_3.jpg" alt="属性" title="タイトル">
</div>

<br>

# Application Architecture

<div align="center">
<img src="https://github.com/hiroyasuakada/dynamic-object-removal-with-unpaired-images/blob/master/demo/GraphicalAbstract1.png" alt="属性" title="タイトル">
</div>

<br>

**Block diagram of our proposed network**

**Training:** In the cycle process, for the given images with a human in
domain X and images without a human in domain Y , the generator G tries to create non-occluded images of G(x) that are
indistinguishable from the images in Y and the discriminator Dy tries to distinguish the fake from the real images, where
adversarial loss is calculated. Then, the generator F tries to generate indistinguishable images of F(y) from the images
in X, where cycle consistency loss is calculated (and vice versa). In the mask process, mask images are extracted from
the ones in X and used to cut out the human-shaped area of both images in X and G(x), yielding images of Xmask and
G(x)mask. Then, we compute MSE loss between the images of Xmask and the corresponding G(x)mask images. 

**Testing:** The generator G is used to translate occluded images into realistic static images without the farmer to evaluate the quality
of the generated images.

<br>

# Usage

## 1. 

## 2. 

# References
