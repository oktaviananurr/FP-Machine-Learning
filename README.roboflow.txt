
New Mod - v1 2024-09-26 7:10am
==============================

This dataset was exported via roboflow.com on September 26, 2024 at 7:13 AM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand and search unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

For state of the art Computer Vision training notebooks you can use with this dataset,
visit https://github.com/roboflow/notebooks

To find over 100k other datasets and pre-trained models, visit https://universe.roboflow.com

The dataset includes 2699 images.
Skin-Conditions are annotated in YOLO v5 PyTorch format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 512x512 (Stretch)

The following augmentation was applied to create 2 versions of each source image:
* 50% probability of horizontal flip
* Random rotation of between -17 and +17 degrees


