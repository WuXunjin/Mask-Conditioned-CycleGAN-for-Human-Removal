# Mask-Conditioned-CycleGAN-for-Human-Removal

# Description

GAN-based Application to remove objects (humans) in images and to reconstruct the occluded areas with plausible background.

keywords: dynamic object removal, generative adversarial network, CycleGAN, unpaired images

# Demo

### 1. Taking a single image with a human and a corresponding mask image as inputs (e.g. in an agricultural setting)

<div align="center">
<!-- <img src="https://github.com/hiroyasuakada/dynamic-object-removal-with-unpaired-images/blob/master/demo/input_1.jpg" alt="属性" title="タイトル"> -->
<img src="https://github.com/hiroyasuakada/dynamic-object-removal-with-unpaired-images/blob/master/demo/input_2.jpg" alt="属性" title="タイトル">
<img src="https://github.com/hiroyasuakada/dynamic-object-removal-with-unpaired-images/blob/master/demo/input_3.jpg" alt="属性" title="タイトル">
</div>

<br>

### 2. Generating an image without the human

<div align="center">
<!-- <img src="https://github.com/hiroyasuakada/dynamic-object-removal-with-unpaired-images/blob/master/demo/output_1.jpg" alt="属性" title="タイトル"> -->
<img src="https://github.com/hiroyasuakada/dynamic-object-removal-with-unpaired-images/blob/master/demo/output_2.jpg" alt="属性" title="タイトル">
<img src="https://github.com/hiroyasuakada/dynamic-object-removal-with-unpaired-images/blob/master/demo/output_3.jpg" alt="属性" title="タイトル">
</div>

<br>

# Application Architecture

<div align="center">
<img src="https://github.com/hiroyasuakada/dynamic-object-removal-with-unpaired-images/blob/master/demo/GraphicalAbstract1.png" alt="属性" title="タイトル">
</div>

<div align="center">
Block diagram of our application
</div>

<br>

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

# Dataset

<div align="center">
<img src="https://github.com/hiroyasuakada/dynamic-object-removal-with-unpaired-images/blob/master/demo/domain_X_Y_small.png" alt="属性" title="タイトル">
</div>

<div align="center">
Examples of our dataset: images with a farmer in domain X (left) and images without a farmer in domain Y (right)
</div>

<br>

Our training dataset comprises 77,849 images: 32,705 images with the worker for domain X and 45,144 images without the worker for domain Y. 
In addition, to analyze our system qualitatively and quantitatively, we prepared 527 images without a human in the farm 
and created corresponding synthetic images with a human.

※ Currently, our agricultural dataset is not open to the public.

# Requirements

Tested on ...

- Linux environment
- One or more NVIDIA GPUs
- NVIDIA drivers, CUDA 9.0 toolkit and cuDNN 7.5
- Python 3.6, PyTorch 1.1
- For docker user, please use the [provided Dockerfile](https://github.com/hiroyasuakada/CycleGAN-PyTorch/blob/main/docker_ITC/dockerfile). (highly recommended)

# Usage
## ① Train CycleGAN

### 1. Download this repository

        git clone https://github.com/hiroyasuakada/CycleGAN-PyTorch.git
        cd CycleGAN-PyTorch

### 2. Prepare dataset

        mkdir dataset
        cd dataset

Please put your dataset in `dataset` folder or download public datasets from [here](https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/).

| Example of folder relation | &nbsp;
| :--- | :----------
| dataset
| &boxur;&nbsp; hourse2zebra
| &ensp;&ensp; &boxur;&nbsp;  trainA | image domain A for training
| &ensp;&ensp; &boxur;&nbsp;  trainB | image domain B for training
| &ensp;&ensp; &boxur;&nbsp;  testA | image domain A for testing
| &ensp;&ensp; &boxur;&nbsp;  testB | image domain B for testing

and then move back to `CycleGAN-PyTorch` folder by `cd ..` command.


### 3. Train the model

        python train.py [DATASET NAME]
        
        # for example
        python train.py horse2zebra
        
This will create `logs` folder in which training details and generated images at each epoch during the training will be saved. 
        
If you have multiple GPUs, 

        python train.py horse2zebra --gpu_ids 0 1 --batch_size 4 

If you want to resume training from a certain epoch, (for example, epoch 25)

        python train.py house2zebra --load_epoch 25

The image is supposed to be a square then this will be resized to 256 × 256.

For more information about training setting, please run `python train.py --help`.



### 3. Test the model

        python test.py [DATASET NAME] [LOAD EPOCH]
        
        # for example, if you want to test your model at epoch 25
        python test.py horse2zebra 25
        
This will create `generated_imgs` folder in which you can find generated images from your model.

For more information about testing setting, please run `python test.py --help`.
        
## Reference

- **pytorch-CycleGAN-and-pix2pix[official]**: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix


# References
["Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks"](https://arxiv.org/abs/1703.10593)
