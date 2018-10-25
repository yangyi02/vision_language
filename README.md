# Demo on Vision Language Learning through PADDLE #

<p align="center">
<img src="challenge.png" alt="Visual Question Answering Example" width="600px">
</p>

### Introduction ###

Human has the remarkable capability of grounding language with vision. 
In Artificial Intelligence, visual language grounding such as image captioning and image question answering are very important but challenging research topics. 
Recently, deep learning shows a great success in speech recognition and computer vision. 
We show that with deep neural networks such as Convolutional Neural Nets and Recurrent Neural Nets, we can further embrace computer vision and natural language.

At Baidu IDL, we started this research project in 2014, under the leadership of Baidu's Distinguished Scientist, Wei Xu, with a list of publications below:

* [Deep Captioning with Multimodal Recurrent Neural Networks (m-RNN)](https://arxiv.org/abs/1412.6632)
* [Learning like a Child: Fast Novel Visual Concept Learning from Sentence Descriptions of Images](http://arxiv.org/abs/1504.06692)
* [Are You Talking to a Machine? Dataset and Methods for Multilingual Image Question Answering](http://arxiv.org/abs/1505.05612)
* [CNN-RNN: A Unified Framework for Multi-label Image Classification](http://www.ics.uci.edu/~yyang8/research/cnn-rnn/cnn-rnn-cvpr2016.pdf)
* [ABC-CNN: An Attention Based Convolutional Neural Network for Visual Question Answering](https://arxiv.org/abs/1511.05960)
* [Video Paragraph Captioning Using Hierarchical Recurrent Neural Networks](http://www.ics.uci.edu/~yyang8/research/video-caption/video-caption.pdf)

The rest of contributors are Junhua Mao, Jiang Wang, Zhiheng Huang, Haoyuan Gao, Lei Wang, Kan Chen, Haonan Yu, Yi Yang, et al.  

To use this code, you need to download and compile the latest version of [Paddle](http://deeplearning.baidu.com).

### Directory Explanation ###

* deploy_demo: demo shows caption generation given a pre-trained image captioning model
* train_demo: demo shows training image captioning, image question answering, or multi-task models
* full_code: the full code that trains and evaluates image captioning and image question answering models on large complicated dataset, such as Microsoft COCO.
