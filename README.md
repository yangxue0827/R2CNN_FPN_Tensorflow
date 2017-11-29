# The code is being sorted out and the complete code and instructions are will uploaded soon.Stay tuned.

A Tensorflow implementation of FPN or R2CNN detection framework based on FPN . 
The paper references [R2CNN Rotational Region CNN for Orientation Robust Scene Text Detection](https://arxiv.org/abs/1706.09579) or [Feature Pyramid Networks for Object Detection](https://arxiv.org/abs/1612.03144)

# Configuration Environment
ubuntu + python2 + tensorflow1.2 + cv2 + cuda8.0 + GeForce GTX 1080     
If you want to use cpu, you need to modify the parameters of NMS and IOU functions use_gpu = False    
You can also use docker environment, command: docker push yangxue2docker/tensorflow3_gpu_cv2_sshd:v1.0     

# Make tfrecord   
The data is VOC format, reference [here](sample.xml)     
data path format  
VOCdevkit  
>VOCdevkit_train  
>>Annotation  
>>JPEGImages   

>VOCdevkit_test   
>>Annotation   
>>JPEGImages   

python ./data/io/convert_data_to_tfrecord.py --VOC_dir='***/VOCdevkit/VOCdevkit_train/' --save_name='train' --img_format='.jpg' --dataset='ship'


# Train
1、Configure parameters in ./libs/configs/cfgs.py and modify the project's root directory    
2、Modify ./libs/lable_name_dict.py, corresponding to the number of categories in the configuration file    
3、download pretrain weight([resnet_v1_101_2016_08_28.tar.gz](http://download.tensorflow.org/models/resnet_v1_101_2016_08_28.tar.gz) or [resnet_v1_50_2016_08_28.tar.gz](http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz)) from [here](https://github.com/yangxue0827/models/tree/master/slim), then extract to folder ./data/pretrained_weights    
4、Choose a model(FPN and R2CNN)     
If you want to train FPN:        
>python ./tools/train.py

elif you want to train R2CNN:     
>python ./tools/train1.py

# Test tfrecord     
mkdir test_result    
python ./tools/test.py(test1.py)   

# Test images  
put images in ./tools/inference_image, and mkdir inference_result    
python ./tools/inference.py(inference1.py)   

# eval   
python ./tools/eval.py(eval1.py)

# Summary   
tensorboard --logdir=./output/summary/   
![01](output/summary/fast_rcnn_loss.bmp) 
![02](output/summary/rpn_loss.bmp) 
![03](output/summary/total_loss.bmp) 

# Graph
![04](graph.png) 

# Test results   
![11](tools/test_result/07_horizontal_gt.jpg)   
![12](tools/test_result/07_horizontal_fpn.jpg)   
     
![13](tools/test_result/07_rotate_gt.jpg)   
![14](tools/test_result/07_rotate_fpn.jpg)  

![15](tools/test_result/08_horizontal_gt.jpg)    
![16](tools/test_result/08_horizontal_fpn.jpg)   
     
![17](tools/test_result/08_rotate_gt.jpg)    
![18](tools/test_result/08_rotate_fpn.jpg)     