## DSLR-Quality Photos on Mobile Devices with Deep Convolutional Networks

<br/>

<img src="http://people.ee.ethz.ch/~ihnatova/assets/img/teaser_git.jpg"/>

<br/>

#### 1. Overview [[Paper]](https://arxiv.org/pdf/1704.02470.pdf) [[Project webpage]](http://people.ee.ethz.ch/~ihnatova/) [[Enhancing RAW photos]](https://github.com/aiff22/PyNET) [[Rendering Bokeh Effect]](https://github.com/aiff22/PyNET-Bokeh)

The provided code implements the paper that presents an end-to-end deep learning approach for translating ordinary photos from smartphones into DSLR-quality images. The learned model can be applied to photos of arbitrary resolution, while the methodology itself is generalized to 
any type of digital camera. More visual results can be found [here](http://people.ee.ethz.ch/~ihnatova/#demo).


#### 2. Prerequisites

- Python + Pillow, scipy, numpy, imageio packages
- [TensorFlow 1.x / 2.x](https://www.tensorflow.org/install/) + [CUDA CuDNN](https://developer.nvidia.com/cudnn)
- Nvidia GPU


#### 3. First steps

- Download the pre-trained [VGG-19 model](https://polybox.ethz.ch/index.php/s/7z5bHNg5r5a0g7k) <sup>[Mirror](https://drive.google.com/file/d/0BwOLOmqkYj-jMGRwaUR2UjhSNDQ/view?usp=sharing&resourcekey=0-Ff-0HUQsoKJxZ84trhsHpA)</sup> and put it into `vgg_pretrained/` folder
- Download [DPED dataset](http://people.ee.ethz.ch/~ihnatova/#dataset) (patches for CNN training) and extract it into `dped/` folder.  
<sub>This folder should contain three subolders: `sony/`, `iphone/` and `blackberry/`</sub>

<br/>

#### 4. Train the model

```bash
python train_model.py model=<model>
```

Obligatory parameters:

>```model```: **```iphone```**, **```blackberry```** or **```sony```**

Optional parameters and their default values:

>```batch_size```: **```50```** &nbsp; - &nbsp; batch size [smaller values can lead to unstable training] <br/>
>```train_size```: **```30000```** &nbsp; - &nbsp; the number of training patches randomly loaded each ```eval_step``` iterations <br/>
>```eval_step```: **```1000```** &nbsp; - &nbsp; each ```eval_step``` iterations the model is saved and the training data is reloaded <br/>
>```num_train_iters```: **```20000```** &nbsp; - &nbsp; the number of training iterations <br/>
>```learning_rate```: **```5e-4```** &nbsp; - &nbsp; learning rate <br/>
>```w_content```: **```10```** &nbsp; - &nbsp; the weight of the content loss <br/>
>```w_color```: **```0.5```** &nbsp; - &nbsp; the weight of the color loss <br/>
>```w_texture```: **```1```** &nbsp; - &nbsp; the weight of the texture [adversarial] loss <br/>
>```w_tv```: **```2000```** &nbsp; - &nbsp; the weight of the total variation loss <br/>
>```dped_dir```: **```dped/```** &nbsp; - &nbsp; path to the folder with DPED dataset <br/>
>```vgg_dir```: **```vgg_pretrained/imagenet-vgg-verydeep-19.mat```** &nbsp; - &nbsp; path to the pre-trained VGG-19 network <br/>

Example:

```bash
python train_model.py model=iphone batch_size=50 dped_dir=dped/ w_color=0.7
```

<br/>

#### 5. Test the provided pre-trained models

```bash
python test_model.py model=<model>
```

Obligatory parameters:

>```model```: **```iphone_orig```**, **```blackberry_orig```** or **```sony_orig```**

Optional parameters:

>```test_subset```: **```full```**,**```small```**  &nbsp; - &nbsp; all 29 or only 5 test images will be processed <br/>
>```resolution```: **```orig```**,**```high```**,**```medium```**,**```small```**,**```tiny```** &nbsp; - &nbsp; the resolution of the test images [**```orig```** means original resolution]<br/>
>```use_gpu```: **```true```**,**```false```** &nbsp; - &nbsp; run models on GPU or CPU <br/>
>```dped_dir```: **```dped/```** &nbsp; - &nbsp; path to the folder with DPED dataset <br/>

Example:

```bash
python test_model.py model=iphone_orig test_subset=full resolution=orig use_gpu=true
```

<br/>

#### 6. Test the obtained models

```bash
python test_model.py model=<model>
```

Obligatory parameters:

>```model```: **```iphone```**, **```blackberry```** or **```sony```**

Optional parameters:

>```test_subset```: **```full```**,**```small```**  &nbsp; - &nbsp; all 29 or only 5 test images will be processed <br/>
>```iteration```: **```all```** or **```<number>```**  &nbsp; - &nbsp; get visual results for all iterations or for the specific iteration,  
>&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;**```<number>```** must be a multiple of ```eval_step``` <br/>
>```resolution```: **```orig```**,**```high```**,**```medium```**,**```small```**,**```tiny```** &nbsp; - &nbsp; the resolution of the test 
images [**```orig```** means original resolution]<br/>
>```use_gpu```: **```true```**,**```false```** &nbsp; - &nbsp; run models on GPU or CPU <br/>
>```dped_dir```: **```dped/```** &nbsp; - &nbsp; path to the folder with DPED dataset <br/>  

Example:

```bash
python test_model.py model=iphone iteration=13000 test_subset=full resolution=orig use_gpu=true
```
<br/>

#### 7. Folder structure

>```dped/```              &nbsp; - &nbsp; the folder with the DPED dataset <br/>
>```models/```            &nbsp; - &nbsp; logs and models that are saved during the training process <br/>
>```models_orig/```       &nbsp; - &nbsp; the provided pre-trained models for **```iphone```**, **```sony```** and **```blackberry```** <br/>
>```results/```           &nbsp; - &nbsp; visual results for small image patches that are saved while training <br/>
>```vgg-pretrained/```    &nbsp; - &nbsp; the folder with the pre-trained VGG-19 network <br/>
>```visual_results/```    &nbsp; - &nbsp; processed [enhanced] test images <br/>

>```load_dataset.py```    &nbsp; - &nbsp; python script that loads training data <br/>
>```models.py```          &nbsp; - &nbsp; architecture of the image enhancement [resnet] and adversarial networks <br/>
>```ssim.py```            &nbsp; - &nbsp; implementation of the ssim score <br/>
>```train_model.py```     &nbsp; - &nbsp; implementation of the training procedure <br/>
>```test_model.py```      &nbsp; - &nbsp; applying the pre-trained models to test images <br/>
>```utils.py```           &nbsp; - &nbsp; auxiliary functions <br/>
>```vgg.py```             &nbsp; - &nbsp; loading the pre-trained vgg-19 network <br/>

<br/>

#### 8. Problems and errors

```
What if I get an error: "OOM when allocating tensor with shape [...]"?
```

&nbsp;&nbsp; Your GPU does not have enough memory. If this happens during the training process:

- Decrease the size of the training batch [```batch_size```]. Note however that smaller values can lead to unstable training.

&nbsp;&nbsp; If this happens while testing the models:

- Run the model on CPU (set the parameter ```use_gpu``` to **```false```**). Note that this can take up to 5 minutes per image. <br/>
- Use cropped images, set the parameter ```resolution``` to:

> **```high```**   &nbsp; - &nbsp; center crop of size ```1680x1260``` pixels <br/>
> **```medium```** &nbsp; - &nbsp; center crop of size ```1366x1024``` pixels <br/>
> **```small```** &nbsp; - &nbsp; center crop of size ```1024x768``` pixels <br/>
> **```tiny```** &nbsp; - &nbsp; center crop of size ```800x600``` pixels <br/>

&emsp;&nbsp; The less resolution is - the smaller part of the image will be processed

<br/>

#### 9. Citation

```
@inproceedings{ignatov2017dslr,
  title={DSLR-Quality Photos on Mobile Devices with Deep Convolutional Networks},
  author={Ignatov, Andrey and Kobyshev, Nikolay and Timofte, Radu and Vanhoey, Kenneth and Van Gool, Luc},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={3277--3285},
  year={2017}
}
```


#### 10. Any further questions?

```
Please contact Andrey Ignatov (andrey.ignatoff@gmail.com) for more information
```
