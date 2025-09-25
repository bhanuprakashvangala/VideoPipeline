# Object Detection and Tracking Pipeline

Here we will build a pipeline with two steps, object detection and object
tracking.

The following provides a brief outline on the entire process, and you are
encouraged to find more models that suit your needs.

## Step 1: Object Dectection

To process a video, we have a series of images that each correspond to
a frame of the video. Hence, we will run the image detection model on each image or frame and pass the output of this detection to the next step.

To do so, we provide you with some popular pretrained options:

1. YOLOv8 from Ultralytics
2. facebook/detr-resnet-50 from HuggingFace
3. fasterrcnn_resnet50_fpn from torchvision

You are free to look for other options and use different models than the ones here.
As a good practice, it is recommended to read up on the documentation on how to
use these models.
Here, we provide https://docs.ultralytics.com/modes/predict/, 
https://huggingface.co/docs/transformers/tasks/object_detection, 
and https://docs.pytorch.org/vision/stable/models.html#object-detection.

## Step 2: Object Tracking

For the second step, we want to track an object as it moves through the video.
To do so we will use specialized softwares like the ones given below.

1. ByteTrack : https://github.com/ifzhang/ByteTrack
2. Oc_Sort : https://github.com/noahcao/OC_SORT

You will have to pass the object detections outputted in the previous steps
as inputs to the object tracking model.
Here is a tutorial on how to use ByteTrack https://www.labellerr.com/blog/how-to-implement-bytetrack/ and here is a sample implementation on how to use Oc_Sort https://github.com/PamanGie/ocsort_yolov8/tree/main.

## Assertion

As an example we can have an assertion that objects given the same
id by the tracking model must be within a certain distance or threshold from each other
across frames.
I.e. A person given id A by the tracking model can only move so far across a
frame, if the gap is too much, then the detection by the tracking model is wrong,
or if the object vanishes and returns after a long time, the detection given by
the object detection model is wrong and needs to be rolled back.
It is up to you to define this threshold.

## Dataset

Download the MOT17 dataset from this website: https://motchallenge.net/data/MOT17/.
If you are running the docker container, then copy the dataset into the container
and then unzip the dataset and place it in the video directory as follows:

eg.
```
video/
|
|
--MOT17/
    |
    |
    --test/
    |
    |
    --train/
```

You can choose to reduce the size of the dataset by copying only a subset of the images
by changing the following line in the dockerfile `COPY MOT17/ /MOT17/`.

> Note that the Dockerfile copies a smaller sample of the dataset on line 26. This is only for testing purposes and can be removed if you want to use a different portion of the dataset.

## Sample
In the container, there is a sample notebook that has some sample code on loading the dataset
and loading/intializing some some sample models that we have provided here.

Note that all the given models work given the dependencies of the requirements file and the
dockerfile.