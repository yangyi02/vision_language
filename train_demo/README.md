### Introduction ###

This is a demo illustrating the training of visual language model training. The code is able to train an image captioning model, an image question answering model, or a multi-task model that combines image captioning and image question answering. 

### Image Captiong Training Demo ###

To run the image captioning training demo, simply modify the PADDLE_PATH in demo_cap.sh and 
```bash
bash demo_cap.sh
```

After training, you can visualize the prediction results on
```bash
cache/task/demo_cap/resnet152_pool5_2048_oversample/train_conf/demo_cap.html
```

### Image Question Answering Training Demo ###

To run the image question answering training demo, simply modify the PADDLE_PATH in demo_vqa.sh and
```bash
bash demo_vqa.sh
```

After training, you can visualize the prediction results on
```bash
cache/task/demo_vqa/resnet152_pool5_2048_oversample/train_conf/demo_vqa.html
```

### Multi-task Training Demo ###

Finally to run the multi-task training on both image captioning and image question answering, simply modify the PADDLE_PATH in demo_multitask.sh and
```bash
bash demo_multitask.sh
```

After training, you can visualize the prediction results on
```bash
cache/task/demo_multitask/resnet152_pool5_2048_oversample/train_conf/demo_cap.html
cache/task/demo_multitask/resnet152_pool5_2048_oversample/train_conf/demo_vqa.html
cache/task/demo_multitask/resnet152_pool5_2048_oversample/train_conf/demo_qa.html
```

### Directory Explanation ###

* demo_images: directory holding demo images
* paddle/data_provider/pydata_test.py: paddle python data provider for generating text
* paddle/data_provider/pydata_train.py: paddle python data provider for training
* paddle/conf/test_conf.py: paddle python configure for generating text
* paddle/conf/train_conf.py: paddle python configure for training
* prepare_data: directory holding data preparation functions for image feature extraction and dictionary generation
* show_results: directory holding codes for generating html files to see image captioning and image question answering prediction results
* data_demo_cap.py: a demo of data processing module for image captioning
* data_demo_qa.py: a demo of data processing module for question answering
* data_demo_vqa.py: a demo of data processing module for image question answering
* visual_language.py: main module for running image captioning, image question answering or multi-task 
