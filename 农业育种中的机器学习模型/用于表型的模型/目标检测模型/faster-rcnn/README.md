## Faster-Rcnn：Two-Stage目标检测模型在Pytorch当中的实现
---

Faster R-CNN是一种用于目标检测的深度学习模型，通过在图像中标记出物体的位置来实现对物体的识别。相比传统的目标检测方法，Faster R-CNN在准确率和速度上都有了明显的提升，使得它在农业领域中的应用变得越来越广泛。在本课题中，我们以农田害虫检测为例，展示了如何使用该模型在农业领域中实现害虫检测的应用。接下来，我们将详细介绍模型的环境配置和代码运行流程，帮助读者更好地理解和应用该模型。

<img src="assets/image-20230426221114397.png" alt="image-20230426221114397" style="zoom:50%;" />

## 目录

1. [所需环境 Environment](#所需环境)
4. [文件下载 Download](#文件下载)
5. [预测步骤 How2predict](#预测步骤)
6. [训练步骤 How2train](#训练步骤)
7. [评估步骤 How2eval](#评估步骤)
8. [参考资料 Reference](#Reference)


## 所需环境
torch == 1.2.0

## 文件下载
训练所需的voc_weights_resnet.pth或者voc_weights_vgg.pth以及主干的网络权重可以在百度云下载。  
voc_weights_resnet.pth是resnet为主干特征提取网络用到的；  
voc_weights_vgg.pth是vgg为主干特征提取网络用到的；   
链接: https://pan.baidu.com/s/1S6wG8sEXBeoSec95NZxmlQ      
提取码: 8mgp    

我们提供了一个VOC格式的病虫害检测数据集示例，下载地址如下： 
链接：https://pan.baidu.com/s/1kuWon5NaDr_iWBT0UP0Olw?pwd=zivn 
提取码：zivn  

## 训练步骤
1. 数据集的准备  
**本文使用VOC格式进行训练，训练前需要自己制作好数据集，**    
训练前将标签文件放在VOCdevkit文件夹下的VOC2007文件夹下的Annotation中。   
训练前将图片文件放在VOCdevkit文件夹下的VOC2007文件夹下的JPEGImages中。   

2. 数据集的处理  
在完成数据集的摆放之后，我们需要利用voc_annotation.py获得训练用的2007_train.txt和2007_val.txt。   
修改voc_annotation.py里面的参数。第一次训练可以仅修改classes_path，classes_path用于指向检测类别所对应的txt。   
训练自己的数据集时，可以自己建立一个cls_classes.txt，里面写自己所需要区分的类别。   
model_data/cls_classes.txt文件内容为：      
```python
cat
dog
...
```
修改voc_annotation.py中的classes_path，使其对应cls_classes.txt，并运行voc_annotation.py。  

3. 开始网络训练  
**训练的参数较多，均在train.py中，大家可以在下载库后仔细看注释，其中最重要的部分依然是train.py里的classes_path。**  
**classes_path用于指向检测类别所对应的txt，这个txt和voc_annotation.py里面的txt一样！训练自己的数据集必须要修改！**  
修改完classes_path后就可以运行train.py开始训练了，在训练多个epoch后，权值会生成在logs文件夹中。  

4. 训练结果预测  
训练结果预测需要用到两个文件，分别是frcnn.py和predict.py。在frcnn.py里面修改model_path以及classes_path。  
**model_path指向训练好的权值文件，在logs文件夹里。  
classes_path指向检测类别所对应的txt。**  
完成修改后就可以运行predict.py进行检测了。运行后输入图片路径即可检测。  

## 预测步骤
### a、使用预训练权重
1. 下载完库后解压，在百度网盘下载frcnn_weights.pth，放入model_data，运行predict.py，输入  
```python
img/street.jpg
```
2. 在predict.py里面进行设置可以进行fps测试和video视频检测。  
### b、使用自己训练的权重
1. 按照训练步骤训练。  
2. 在frcnn.py文件里面，在如下部分修改model_path和classes_path使其对应训练好的文件；**model_path对应logs文件夹下面的权值文件，classes_path是model_path对应分的类**。  
```python
_defaults = {
    #--------------------------------------------------------------------------#
    #   使用自己训练好的模型进行预测一定要修改model_path和classes_path！
    #   model_path指向logs文件夹下的权值文件，classes_path指向model_data下的txt
    #   如果出现shape不匹配，同时要注意训练时的model_path和classes_path参数的修改
    #--------------------------------------------------------------------------#
    "model_path"    : 'model_data/voc_weights_resnet.pth',
    "classes_path"  : 'model_data/voc_classes.txt',
    #---------------------------------------------------------------------#
    #   网络的主干特征提取网络，resnet50或者vgg
    #---------------------------------------------------------------------#
    "backbone"      : "resnet50",
    #---------------------------------------------------------------------#
    #   只有得分大于置信度的预测框会被保留下来
    #---------------------------------------------------------------------#
    "confidence"    : 0.5,
    #---------------------------------------------------------------------#
    #   非极大抑制所用到的nms_iou大小
    #---------------------------------------------------------------------#
    "nms_iou"       : 0.3,
    #---------------------------------------------------------------------#
    #   用于指定先验框的大小
    #---------------------------------------------------------------------#
    'anchors_size'  : [8, 16, 32],
    #-------------------------------#
    #   是否使用Cuda
    #   没有GPU可以设置成False
    #-------------------------------#
    "cuda"          : True,
}
```
3. 运行predict.py，输入  
```python
img/street.jpg
```
4. 在predict.py里面进行设置可以进行fps测试和video视频检测。  

## 评估步骤 
1. 本文使用VOC格式进行评估。  
2. 如果在训练前已经运行过voc_annotation.py文件，代码会自动将数据集划分成训练集、验证集和测试集。如果想要修改测试集的比例，可以修改voc_annotation.py文件下的trainval_percent。trainval_percent用于指定(训练集+验证集)与测试集的比例，默认情况下 (训练集+验证集):测试集 = 9:1。train_percent用于指定(训练集+验证集)中训练集与验证集的比例，默认情况下 训练集:验证集 = 9:1。
3. 利用voc_annotation.py划分测试集后，前往get_map.py文件修改classes_path，classes_path用于指向检测类别所对应的txt，这个txt和训练时的txt一样。评估自己的数据集必须要修改。
4. 在frcnn.py里面修改model_path以及classes_path。**model_path指向训练好的权值文件，在logs文件夹里。classes_path指向检测类别所对应的txt。**  
5. 运行get_map.py即可获得评估结果，评估结果会保存在map_out文件夹中。

## Reference

https://github.com/bubbliiiing/faster-rcnn-pytorch
https://github.com/chenyuntc/simple-faster-rcnn-pytorch  
https://github.com/eriklindernoren/PyTorch-YOLOv3  
https://github.com/BobLiu20/YOLOv3_PyTorch  