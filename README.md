# Object Detection with YOLO

This week, our focus will be on Convolutional Neural Networks (CNNs), specifically the YOLO (You Only Look Once) model architecture. YOLO is a significant leap in complexity, particularly known for its effectiveness in object detection tasks. If you encounter any difficulties in understanding the code or concepts, please do not hesitate to book office hours with Bes. Understanding YOLO is crucial as it stands as one of the most popular object detection architectures in use today.

Goal of the week
Your task for the week is to train a YOLO model from scratch. The objective is to detect and identify numbers in images from Theera, particularly focusing on MNIST digits. You need to accurately determine the bounding box and identify the digit it encloses. An example of the expected input and output is provided below for reference.

https://cdn.mlx.institute/assets/cortex.07.png

To clarify, by the end of the week, you should have a model deployed to Theera. When an image with one or more digits is input, your model/server should return an image highlighting each digit with its respective bounding box.

We recommend the following approach:

Generate a synthetic dataset using MNIST digits.
Start by detecting a single number in an image.
Develop the model to predict only one bounding box initially.
Train this simplified model and iteratively improve it.
Given the increased complexity and volume of coding required for this challenge, we will provide code snippets and guidance as the week progresses.

Learning outcomes
At a high level by the end of this week, you should know:

Understanding of YOLOv1 Model Architecture
CNN Fundamentals
Data Preparation and Synthetic Dataset Creation
Object Detection Techniques
References
There are many papers and articles about YOLO but we recommend starting with the original paper.

You Only Look Once: Unified, Real-Time Object Detection
Paper: https://arxiv.org/pdf/1506.02640.pdf
