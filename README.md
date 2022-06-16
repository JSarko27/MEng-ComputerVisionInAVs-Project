# MEng-AV-Project
Final Year individual project on how computer vision can be implemented in autonomous vehicles with object detection
Object detection was achieved by labelling my own dataset and training on YOLOv4-tiny network

# Thesis for Project
For more details regarding the process and findings from this project, see the **thesis.pdf** file.

# Installing YOLOv4
To install YOLOv4 locally, a system with an NVIDIA GPU is required, as well as the following software prerequisites:

Microsoft Visual Studio Community 2019

Git 2.35

Microsoft Powershell

CUDA Computing Toolkit 11.6

cuDNN 8.3.2

CMake 3.22.2

OpenCV 4.5.2

OpenCV-contrib 4.5.2

Anaconda

[MinGW](https://sourceforge.net/projects/mingw/)

The first step to be taken is to ensure that your system has the required software development environment required to compile programs later during the installation phase. This is achieved by installing the [MinGW](https://sourceforge.net/projects/mingw/) base compiler first.

Once the software is installed, to install the necessary compiler, only **mingw32-base** & **mingw32-gcc-g++** were selected before proceeding with the install, however there is no harm in selecting more packages, as long as the aforementioned packages are selected.

The next steps can be found on the original [YOLOv4](https://github.com/AlexeyAB/darknet#how-to-compile-on-windows-using-vcpkg) repository. When installing the packages, ensure environment variables are designated to the correct directories. Ensure the option "**-EnableOPENCV_CUDA**" is typed in powershell before running the vcpkg install in command line, this will ensure that OpenCV's GPU capabilities can be fully utilised.

## **Building OpenCV with CUDA support** 

I personally had issues while using the vcpkg option to install OpenCV with CUDA support. A link to a useful step by step video to install OpenCV with CUDA support is provided below.

**Note: When creating a python environment, ensure the same environment is used throughout the project.**

To build a darknet project, open the CMake GUI and specify the address of the source code and the folder for the project to be built to the darknet folder previously specified. Then click "Configure" and set the following parameters:

Generator for Project: Visual Studio 16 2019

Optional Platform: x64

Leave the rest as default and click "Finish".

Once the initial configuration is complete, check the CUDA Architecture of your system [**here**](https://developer.nvidia.com/cuda-gpus). Then head back to the CMake GUI and click the drop down of the "CMAKE" section and change the architecture to the architecture of your system. If you have an archiecture of 8.6, ensure you enter 86.

Then click the drop down of "ENABLE" and ensure the following options are selected:
- ENABLE_CUDA
- ENABLE_CUDNN
- ENABLE_OPENCV
- ENABLE_VCPKG_INTEGRATION

Then generate the project by clicking "Generate".

Visual Studio should open with a Darknet project. Along the top of the application next to the "x64" dropdown, change "Debug" to "Release". Then select the "Build" ribbon and enter the "Configuration Manager". Select the tickbox corresponding with "INSTALL" under the "Build" column. Close the Configuration Manager and build the project by clicking "Build Solution" under the "Build" ribbon.

Here are some good step by step video tutorials to install YOLO and OpenCV if you run into any issues with the vcpkg method.

[YOLO Install](https://www.youtube.com/watch?v=WK_2bpWj35A)

[OpenCV with CUDA](https://www.youtube.com/watch?v=HsuKxjQhFU0&t=1148s)

# Creating a Dataset

The purpose of creating a dataset is to train the YOLO network to identify objects you desire it to detect. For this project, we intend to detect the following:

- Pedestrians
- Traffic Lights
- Stop Signs
- Vehicles
- School Crossing sign
- 20mph speed sign
- 30mph speed sign
- 40mph speed sign
- 50mph speed sign
- 60mph speed sign
- 70mph speed sign
- 80mph speed sign

For the model to perform well, we need a high volume of images that are correctly labelled with instances of the objects previously mentioned. The more instances of objects passed through the network for training, the better the performance we can expect. There are some examples of Traffic Sign datasets available online to choose from, namely [Road Sign Detection from Kaggle](https://www.kaggle.com/datasets/andrewmvd/road-sign-detection), [GTSRB - German Traffic Sign Recognition Benchmark](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign) and [Mapillary](https://www.mapillary.com/dataset/trafficsign). 

However for the specificity of the task at hand, I went out of my way to produce a model with open source images. A few sources of images include [Unsplash](https://unsplash.com/), [Google Images](https://www.google.com/imghp?hl=EN) and [Pixabay](https://pixabay.com/). A python script (download_from_google.py) from [**this repository**](https://github.com/haroonshakeel/simple_image_download) was used to mass download images from Google Images and the [Image Downloader](https://chrome.google.com/webstore/detail/image-downloader/cnpniohnfphhjihaiiggeabnkjhpaldj) google extension was also used to mass download from the open source image websites. Ensure all the images are stored in a single folder and are **ALL IN ".jpg" FORMAT.**

## Augmenting the images

To provide more context to the images within the dataset, python scripts were developed to implement different methods of image augmentation. This was achieved using the ImgAug python library, which can be obtained by entering the following command in an Anaconda prompt (ensure the correct python environment is selected):

pip install imgaug

The scripts can be found in the ___ folder.

The ImgAug library has functions that allow you perform rotation and many more augmentation methods. In this project, we tried to emulate potential scenarios in which a driver may have to encounter the objects to be detected, such as rainy/foggy/snowy conditions. These weather conditions, as well as different levels of zoom, brightness and more methods were applied to the set of images with the use of several python scripts. This allowed each generated picture to follow a naming scheme which made it easier to find particular images if needed.

## Labelling the dataset

[LabelImg](https://github.com/tzutalin/labelImg/releases/tag/v1.8.1)

The LabelImg software specified above is the software used to annotate the images within the dataset. Extract the folder to the same directory as the dataset. A .txt file containing the names of the classes to be annotated must be provided in the "data" folder of the windows_v1.8.1 folder, which will allow you to specify what objects are being annotated in each image. 

Open the software and click "Open" to navigate to the folder of the dataset. This will load up the first image of the dataset. Then ensure the files to be generated are of the **YOLO** type, which can be done by clicking the box on the left of the GUI named "Pascal VOC" several times until "YOLO" appears.

To annotate an image, click the "Create/nRectBox" and draw a box over the object to be detected which is present in the image. This process can be repeated multiple times if there are multiple instances of objects to be detected. Once all objects are highlighted, click "Save" and ensure the text file is to be saved in the same directory as the dataset. Repeat the process until all images are annotated. Once this is done, you should have a dataset with a .jpg file with a corrsponding .txt file.
