# tf-object-detection
This repository is just collecting the documents and guide how to install,train and deploy object detection model based on Tensorflow object detection API locally.

## Install object detection model
You can use pip to install tensorflow,or you can refer to [Installation of tensorflow](https://www.tensorflow.org/install/install_linux) and use other methods to install
```bash
    sudo pip install tensorflow     #for CPU version
    sudo pip install tensorflow-gpu #for GPU version
```
Please refer to [Install models of tensorflow](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/installation.md) and install object detection model.

## Prepare PASCAL VOC TFRecord files
```bash
cd <models/object-detection>
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
tar -xvf VOCtrainval_11-May-2007.tar
python create_pascal_tf_record.py --data_dir=VOCdevkit \
    --year=VOC2007 --set=train --output_path=pascal_train.record

python create_pascal_tf_record.py --data_dir=VOCdevkit \
        --year=VOC2007 --set=val --output_path=pascal_val.record
```

## Download the pre-trained model
We will use pre-trained model to fine tune with new objects dataset which could save our time and get good result.
```
wget http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_11_06_2017.tar.gz
tar -xvf faster_rcnn_resnet101_coco_11_06_2017.tar.gz
```

## Modify configuration file for training
I just use faster_rcnn_resnet101_voc07.config 
```bash
#<from models/object_detection/>
cp object_detection/samples/configs/faster_rcnn_resnet101_voc07.config .
vim faster_rcnn_resnet101_voc07.config
```

## Modify the configuration file
```
    model {
      faster_rcnn {
        num_classes: 20
        image_resizer {
          keep_aspect_ratio_resizer {
            min_dimension: 600
            max_dimension: 1024
          }
        }
        feature_extractor {
          type: 'faster_rcnn_resnet101'
          first_stage_features_stride: 16
        }
    
    train_config: {
      batch_size: 1
      optimizer {
        momentum_optimizer: {
          learning_rate: {
            manual_step_learning_rate {
              initial_learning_rate: 0.0001
              schedule {
                step: 0
                learning_rate: .0001
              }
              schedule {
                step: 5000
                learning_rate: .00001
              }
              schedule {
                step: 7000
                learning_rate: .000001
              }
            }
          }
          momentum_optimizer_value: 0.9
        }
        use_moving_average: false
      }
      gradient_clipping_by_norm: 10.0
      fine_tune_checkpoint: "faster_rcnn_resnet101_coco_11_06_2017/model.ckpt"
      from_detection_checkpoint: true
      num_steps: 10000
      data_augmentation_options {
        random_horizontal_flip {
        }
      }
    }
    
    train_input_reader: {
      tf_record_input_reader {
        input_path: "./pascal_train.record"
      }
      label_map_path: "./data/pascal_label_map.pbtxt"
    }
    
    eval_config: {
      num_examples: 4952
    }
    
    eval_input_reader: {
      tf_record_input_reader {
        input_path: "./pascal_val.record"
      }
      label_map_path: ".data/pascal_label_map.pbtxt"
      shuffle: false
      num_readers: 1
    }
```
You can to modify the some paramters in the configuration file accroding to your real dataset.
1. label_map_path
2. input_path
3. num_steps
4. learning_rate
5. num_classes 

## Train

```bash
python object_detection/train.py \
    --logtostderr \
    --pipeline_config_path=./faster_rcnn_resnet101_voc07.config \
    --train_dir=./trained-result
```
Check the checkpoint files
```bash
ls trained-result
checkpoint graph.pbtxt model.ckpt-504.data-00000-of-00001 model.ckpt-504.index model.ckpt-504.meta
```

## Freeze model file
```bash
python export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path ./faster_rcnn_resnet101_voc07.config \
    --trained_checkpoint_prefix ./trained-result/model.ckpt-<step number> \
    --output_directory ./freezed-graph
```
* Check the generated model file.
```bash
ls freezed-graph
checkpoint  frozen_inference_graph.pb  model.ckpt.data-00000-of-00001  model.ckpt.index  model.ckpt.meta  saved_model
```
And the ** frozen_inference_graph.pb** file we cloud deploy to the application.


## Run with retrain model file.



## Ref
1. [RUN FASTER RCNN USING TENSORFLOW DETECTION API](https://data-sci.info/2017/06/27/run-faster-rcnn-tensorflow-detection-api/)
2. [Install models of tensorflow](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/installation.md)
3. [Detection model zoo ](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/detection_model_zoo.md)
