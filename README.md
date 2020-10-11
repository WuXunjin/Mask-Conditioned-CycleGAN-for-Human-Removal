# Mask-Conditioned-CycleGAN-for-Human-Removal

# Description

Mask-Conditioned CycleGAN to remove objects (humans) in images and to reconstruct the occluded areas with plausible background.

You can train Mask-Conditioned-CycleGAN **with your custom dataset**.

If you want to use normal CycleGAN implementation, please see [this repository](https://github.com/hiroyasuakada/CycleGAN-PyTorch)

keywords: dynamic object removal, generative adversarial network, CycleGAN, unpaired images

# Demo

### 1. Taking a single image with a human as an input (e.g. in an agricultural setting)

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
in X, where cycle consistency loss is calculated (and vice versa). In the mask process, mask images are used 
to cut out the human-shaped area of both images in X and G(x), yielding images of Xmask and
G(x)mask. Then, we compute MSE loss between the images of Xmask and the corresponding G(x)mask images. 

**Testing:** The generator G is used to translate occluded images into realistic static images without the farmer to evaluate the quality
of the generated images.

<br>

# Our Dataset

<div align="center">
<img src="https://github.com/hiroyasuakada/dynamic-object-removal-with-unpaired-images/blob/master/demo/domain_X_Y_small.png" alt="属性" title="タイトル">
</div>

<div align="center">
Examples of our dataset: images with a farmer in domain X (left) and images without a farmer in domain Y (right).
</div>

<br>

Our training dataset comprises 110,554 images: 32,705 images with the worker for domain X (and the same number of mask images) and 45,144 images without the worker for domain Y. 
In addition, to analyze our system qualitatively and quantitatively, we prepared 527 images without a human in the farm 
and created corresponding synthetic images with a human.

※ Currently, our agricultural dataset is not open to the public.

# Requirements

Tested on ...

- Linux environment
- One or more NVIDIA GPUs
- NVIDIA drivers, CUDA 9.0 toolkit and cuDNN 7.5
- Python 3.6, PyTorch 1.1
- For docker user, please use the [provided Dockerfile](https://github.com/hiroyasuakada/Mask-Conditioned-CycleGAN-for-Human-Removal/blob/master/docker_ITC/dockerfile). (highly recommended)

# Usage
## Train Mask-Conditioned CycleGAN with your custom dataset

### 1. Download this repository

        git clone https://github.com/hiroyasuakada/Mask-Conditioned-CycleGAN-for-Human-Removal.git
        cd Mask-Conditioned-CycleGAN-for-Human-Removal
### 2. Prepare dataset

        mkdir dataset
        cd dataset

Please put your dataset in `dataset` folder.

If you don't have mask images, please use [this application](https://github.com/hiroyasuakada/Mask-RCNN-Detectron2-for-Human-Extraction) 
to get them. (Note that 'cropped_figure' is equivalent to our mask image.)

| Example of folder relation | &nbsp;
| :--- | :----------
| dataset
| &boxur;&nbsp; [YOUR DATASET NAME]
| &ensp;&ensp; &boxur;&nbsp;  trainA | training images with a human in domain A
| &ensp;&ensp; &boxur;&nbsp;  trainB | training images without a human in domain B
| &ensp;&ensp; &boxur;&nbsp;  trainA_mask | mask images in domain A
| &ensp;&ensp; &boxur;&nbsp;  testA | testing images with a human in domain A
| &ensp;&ensp; &boxur;&nbsp;  testB | testing images without a human in domain B

and then move back to `Mask-Conditioned-CycleGAN-for-Human-Removal` folder by `cd ..` command.


### 3. Train the model

        python train.py [YOUR DATASET NAME]
        
This will create `logs` folder in which training details and generated images at each epoch during the training will be saved. 
        
If you have multiple GPUs, 

        python train.py [YOUR DATASET NAME] --gpu_ids 0 1 --batch_size 4 

If you want to resume training from a certain epoch, (for example, epoch 25)

        python train.py [YOUR DATASET NAME] --load_epoch 25

For more information about training setting, please run `python train.py --help`.


### 4. Test the model

        python test.py [DATASET NAME] [LOAD EPOCH]
        
        # for example, if you want to test your model at epoch 25
        python test.py [DATASET NAME] 25
        
This will create `generated_imgs` folder in which you can find generated images from your model.

For more information about testing setting, please run `python test.py --help`.
        
## Reference

- [**"Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks"**](https://arxiv.org/abs/1703.10593)

