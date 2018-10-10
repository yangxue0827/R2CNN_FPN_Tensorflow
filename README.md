# R2CNN: Rotational Region CNN for Orientation Robust Scene Detection

## Recommend improved code： https://github.com/DetectionTeamUCAS     

A Tensorflow implementation of FPN or R2CNN detection framework based on [FPN](https://github.com/yangxue0827/FPN_Tensorflow).  
You can refer to the papers [R2CNN Rotational Region CNN for Orientation Robust Scene Text Detection](https://arxiv.org/abs/1706.09579) or [Feature Pyramid Networks for Object Detection](https://arxiv.org/abs/1612.03144)    
Other rotation detection method reference [R-DFPN](https://github.com/yangxue0827/R-DFPN_FPN_Tensorflow), [RRPN](https://github.com/yangJirui/RRPN_FPN_Tensorflow) and [R2CNN_HEAD](https://github.com/yangxue0827/R2CNN_HEAD_FPN_Tensorflow)       
If useful to you, please star to support my work. Thanks.    

## Citation
Some relevant achievements based on this code.     

    @article{[yang2018position](https://ieeexplore.ieee.org/document/8464244),
		title={Position Detection and Direction Prediction for Arbitrary-Oriented Ships via Multitask Rotation Region Convolutional Neural Network},
		author={Yang, Xue and Sun, Hao and Sun, Xian and  Yan, Menglong and Guo, Zhi and Fu, Kun},
		journal={IEEE Access},
		volume={6},
		pages={50839-50849},
		year={2018},
		publisher={IEEE}
	}
    
    @article{[yang2018r-dfpn](http://www.mdpi.com/2072-4292/10/1/132),
		title={Automatic ship detection in remote sensing images from google earth of complex scenes based on multiscale rotation dense feature pyramid networks},
		author={Yang, Xue and Sun, Hao and Fu, Kun and Yang, Jirui and Sun, Xian and Yan, Menglong and Guo, Zhi},
		journal={Remote Sensing},
		volume={10},
		number={1},
		pages={132},
		year={2018},
		publisher={Multidisciplinary Digital Publishing Institute}
	} 

## Configuration Environment
ubuntu(Encoding problems may occur on windows) + python2 + tensorflow1.2 + cv2 + cuda8.0 + GeForce GTX 1080     
If you want to use cpu, you need to modify the parameters of NMS and IOU functions use_gpu = False  in cfgs.py     
You can also use docker environment, command: docker pull yangxue2docker/tensorflow3_gpu_cv2_sshd:v1.0    

## Installation      
  Clone the repository    
  ```Shell    
  git clone https://github.com/yangxue0827/R2CNN_FPN_Tensorflow.git    
  ```     

## Make tfrecord   
The data is VOC format, reference [here](sample.xml)       
Data path format  ($R2CNN_ROOT/data/io/divide_data.py)    
```
├── VOCdevkit
│   ├── VOCdevkit_train
│       ├── Annotation
│       ├── JPEGImages
│    ├── VOCdevkit_test
│       ├── Annotation
│       ├── JPEGImages
```  

Clone the repository    
  ```Shell    
  cd $R2CNN_ROOT/data/io/  
  python convert_data_to_tfrecord.py --VOC_dir='***/VOCdevkit/VOCdevkit_train/' --save_name='train' --img_format='.jpg' --dataset='ship'
       
  ``` 

## Compile
```  
cd $PATH_ROOT/libs/box_utils/
python setup.py build_ext --inplace
```

##Demo   
1、Unzip the weight $R2CNN_ROOT/output/res101_trained_weights/*.rar    
2、put images in $R2CNN_ROOT/tools/inference_image   
3、Configure parameters in $R2CNN_ROOT/libs/configs/cfgs.py and modify the project's root directory    
4、     
  ```Shell    
  cd $R2CNN_ROOT/tools      
  ```    
5、image slice        
  ```Shell    
  python inference1.py   
  ```      
      
6、large image      
  ```Shell    
  cd $FPN_ROOT/tools
  python demo1.py --src_folder=.\demo_src --des_folder=.\demo_des         
  ```   

## Train   
1、Modify $R2CNN_ROOT/libs/lable_name_dict/***_dict.py, corresponding to the number of categories in the configuration file    
2、download pretrain weight([resnet_v1_101_2016_08_28.tar.gz](http://download.tensorflow.org/models/resnet_v1_101_2016_08_28.tar.gz) or [resnet_v1_50_2016_08_28.tar.gz](http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz)) from [here](https://github.com/yangxue0827/models/tree/master/slim), then extract to folder $R2CNN_ROOT/data/pretrained_weights    
3、  
  ```Shell    
  cd $R2CNN_ROOT/tools      
  ``` 
4、Choose a model([FPN](https://github.com/yangxue0827/FPN_Tensorflow) or R2CNN))     
If you want to train [FPN](https://github.com/yangxue0827/FPN_Tensorflow) :        
  ```Shell    
  python train.py   
  ```      

elif you want to train R2CNN:  
   ```Shell    
  python train1.py   
  ``` 

## Test tfrecord     
  ```Shell    
  cd $R2CNN_ROOT/tools   
  python test.py(test1.py)   
  ```    

## eval(Not recommended, Please refer [here](https://github.com/DetectionTeamUCAS)    
  ```Shell    
  cd $R2CNN_ROOT/tools   
  python eval.py(eval1.py)  
  ```  

## Summary    
  ```Shell    
  tensorboard --logdir=$R2CNN_ROOT/output/res101_summary/ 
  ```     
![01](output/res101_summary/fast_rcnn_loss.bmp) 
![02](output/res101_summary/rpn_loss.bmp) 
![03](output/res101_summary/total_loss.bmp) 

## Graph
![04](graph.png) 

## icdar2015 test results      
![19](tools/test_result/img_108.jpg_horizontal_fpn.jpg)     
![20](tools/test_result/img_108.jpg_rotate_fpn.jpg)    

![21](tools/test_result/img_51.jpg_horizontal_fpn.jpg)     
![22](tools/test_result/img_51.jpg_rotate_fpn.jpg)    

![23](tools/test_result/img_403.jpg_horizontal_fpn.jpg)     
![24](tools/test_result/img_403.jpg_rotate_fpn.jpg)    

## Test results     
![11](tools/test_result/07_horizontal_gt.jpg)   
![12](tools/test_result/07_horizontal_fpn.jpg)   
     
![13](tools/test_result/07_rotate_gt.jpg)   
![14](tools/test_result/07_rotate_fpn.jpg)  

![15](tools/test_result/08_horizontal_gt.jpg)    
![16](tools/test_result/08_horizontal_fpn.jpg)   
     
![17](tools/test_result/08_rotate_gt.jpg)    
![18](tools/test_result/08_rotate_fpn.jpg)     