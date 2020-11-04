# Visdrone无人机图像目标检测

本仓库是人工智能课程的课程作业仓库，主要是完成无人机图像目标检测的任务，我们对visdrone数据集进行了处理，在yolo和ssd两种框架下进行了训练和测试，并编写demo用于实时的无人机图像目标检测。

> 小组成员：宋晨明、王有发、刘竹风、王伟业
>
> 仅用于西安交通大学人工智能大作业
>
> 模型及数据集下载地址：[百度网盘](https://pan.baidu.com/s/1voGZhYyvEHal_uRPxUownQ) 密码：c7z2

## requirements

ssd: pytorch1.4

yolo:pytorch1.0 tensorflow1.14

## 结构

```
visdrone_detection
├─ readme.md
├─ ssd
├─ yolo
└─ 人工智能大作业_流程.md
```

## 数据集

本作业使用的数据集是VisDrone数据集，包含了10个类（即行人、人、汽车、面包车、公共汽车、卡车、汽车、自行车、遮阳三轮车和三轮车），数据集下载地址为：[VisDrone数据集下载地址](http://aiskyeye.com/challenge/object-detection/)

由于本次作业我们使用SSD和YOLO框架来完成目标检测的任务，需要对原先的数据标注格式进行处理，数据集原始的标注形式为xml文件，我们将标注转化为了常用的voc形式和yolo形式，您可以使用代码中的utils下的代码文件自行处理数据，也可以联系我们获取。

## 训练

### ssd训练

首先请cd到`ssd`目录下，调试`visdrone_train.py`下数据集和配置文件的路径信息，本作业的配置文件为`configs/vgg_ssd300_visdrone0413.yaml`，执行：

```
python visdrone_train.py 
```

训练好的模型文件将会保存在`outputs`目录下

### yolo训练

首先请cd到`yolo`目录下，生成cfg文件和custom.data

```
python train.py --model_def config/yolov3-custom.cfg --data_config config/custom.data--pretrained_weights weights/darknet53.conv.74
```

训练好的模型文件将会保存在`checkpoints`目录下

## 测试

### ssd测试

首先请cd到`ssd`目录下, 执行

```
python visdrone_test.py
python visdrone_demo.py
```

### yolo测试

首先请cd到`yolo`目录下, 执行

```
python test.py --weights_path weights/yolov3.weights
python3 detect.py --image_folder data/samples/
```

## 结果

* mAP指标

  ```
  # yolo
  +-------+-----------------+---------+
  | Index | Class name      | AP      |
  +-------+-----------------+---------+
  | 0     | pedestrian      | 0.08822 |
  | 1     | people          | 0.02340 |
  | 2     | bicycle         | 0.00165 |
  | 3     | car             | 0.43279 |
  | 4     | van             | 0.07407 |
  | 5     | truck           | 0.07747 |
  | 6     | tricycle        | 0.00995 |
  | 7     | awning-tricycle | 0.01727 |
  | 8     | bus             | 0.25008 |
  | 9     | motor           | 0.05405 |
  | 10    | others          | 0.00366 |
  +-------+-----------------+---------+
  ---- mAP 0.09387386786609676
  # ssd
  2020-11-04 14:56:20,698 SSD.inference INFO: mAP: 0.1524
  pedestrian      : 0.1170
  people          : 0.0909
  bicycle         : 0.0909
  car             : 0.4377
  van             : 0.1740
  truck           : 0.2258
  tricycle        : 0.1048
  awning-tricycle : 0.0413
  bus             : 0.3754
  motor           : 0.0800
  others          : 0.0909
  ```

* FPS

  ```
  # yolo
  fps: 19.17227602398063
  83.97542357444763s / 1610imgs
  # ssd
  FPS:64.44748916337679
  24.98157835006714s / 1610imgs
  ```

* 部分结果

  **yolo**

  ![0000006_00159_d_0000001](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/typoraimgs/0000006_00159_d_0000001.png)

  **ssd**

  ![0000006_00159_d_0000001](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/typoraimgs/0000006_00159_d_0000001.jpg)

## References

* [目录生成工具](http://dir.yardtea.cc/)
* [YOLOv3: An Incremental Improvement](https://github.com/eriklindernoren/PyTorch-YOLOv3)

* [SSD: Single Shot MultiBox Object Detector, in PyTorch](https://github.com/amdegroot/ssd.pytorch)



## TODO

好多要做的，完善流程