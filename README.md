# MEng-AV-Project
Final Year individual project on how computer vision can be implemented in autonomous vehicles with object detection
Object detection was achieved by labelling my own dataset and training on YOLOv4-tiny network.
![image](https://user-images.githubusercontent.com/99783917/174675282-7b1216f2-ac2f-4a6d-9889-e7bbba2f1f0e.png)


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

However for the specificity of the task at hand, I went out of my way to produce a model with open source images. A few sources of images include [Unsplash](https://unsplash.com/), [Google Images](https://www.google.com/imghp?hl=EN) and [Pixabay](https://pixabay.com/). A python script (download_from_google.py) from [**this repository**](https://github.com/haroonshakeel/simple_image_download) was used to mass download images from Google Images and the [Image Downloader](https://chrome.google.com/webstore/detail/image-downloader/cnpniohnfphhjihaiiggeabnkjhpaldj) google extension was also used to mass download from the open source image websites.

There are pre-labelled open source image datasets available on [Google](https://storage.googleapis.com/openimages/web/visualizer/index.html?set=train&type=detection&c=%2Fm%2F0k4j) for any classes you may need extra images for. Images were mass downloaded using [OIDv4 Toolkit](https://github.com/theAIGuysCode/OIDv4_ToolKit), particularly for extra traffic lights and vehicle representation in the dataset. More guidance can be found in [this Youtube Tutorial](https://www.youtube.com/watch?v=mmj3nxGT2YQ&list=PLKHYJbyeQ1a3tMm-Wm6YLRzfW1UmwdUIN&index=2). 

Ensure all the images are stored in a single folder and are **ALL IN ".jpg" FORMAT** for the dataset.

## Augmenting the images

To provide more context to the images within the dataset, python scripts were developed to implement different methods of image augmentation. This was achieved using the ImgAug python library, which can be obtained by entering the following command in an Anaconda prompt (ensure the correct python environment is selected):

pip install imgaug

The scripts can be found in the ___ folder.

The ImgAug library has functions that allow you perform rotation and many more augmentation methods. In this project, we tried to emulate potential scenarios in which a driver may have to encounter the objects to be detected, such as rainy/foggy/snowy conditions. These weather conditions, as well as different levels of zoom, brightness and more methods were applied to the set of images with the use of several python scripts. This allowed each generated picture to follow a naming scheme which made it easier to find particular images if needed.

## Labelling the dataset

[LabelImg](https://github.com/tzutalin/labelImg/releases/tag/v1.8.1)

The LabelImg software specified above is the software used to annotate the images within the dataset. Extract the folder to the same directory as the dataset. A .txt file containing the names of the classes to be annotated must be provided in the "data" folder of the windows_v1.8.1 folder, which will allow you to specify what objects are being annotated in each image. Ensure that each class name is written on a new line in the text file, as each line specifies the name of a new class. 

Open the software and click "Open" to navigate to the folder of the dataset. This will load up the first image of the dataset. Then ensure the files to be generated are of the **YOLO** type, which can be done by clicking the box on the left of the GUI named "Pascal VOC" several times until "YOLO" appears.

To annotate an image, click the "Create/nRectBox" and draw a box over the object to be detected which is present in the image. This process can be repeated multiple times if there are multiple instances of objects to be detected. Once all objects are highlighted, click "Save" and ensure the text file is to be saved in the same directory as the dataset. Repeat the process until all images are annotated. Once this is done, you should have a dataset with a .jpg file with a corrsponding .txt file.

## Increasing the size of the dataset

[Roboflow](https://roboflow.com/)

[Roboflow](https://roboflow.com/) is a good website to use to increase the size and add even more augmentation methods to your labelled dataset. You can also use it to manage and observe the number of labelled instances for each class in your dataset. Even after augmentation, Roboflow will generate YOLO files that correspond with the updated coordinates of where the object is present in an image, which is why it is important to label your dataset first before using this tool.

## Training the dataset

In the Darknet directory, open the **Data** folder and create a new folder called **"obj"**. Then copy and paste the dataset into this folder. In the **Data** folder again, copy and paste the **"coco.names"** and **"coco.data"** files and rename them to **obj.names** and **obj.data** respectively. In the "obj.names" file, replace the contents of the file with the contents of the text file placed in the data folder used for labelling. This will line up the class numbers with the names of the classes for training. In the "obj.data" file you must specify the parameters such as the number of classes in the dataset and the directories where the text files for training and validation (validation not mandatory) are. Finally in the darknet directory, create a folder named **"Backup"**, this is where we will store our trained weights.

We must establish the VRAM requirement of your system. This will allow us to determine what version of YOLO you can train your machine on, otherwise you will encounter memory issues.
Back in the darknet directory, open the **"cfg"** folder and look for the **"yolov4-tiny-custom.cfg** or **"yolov4-custom.cfg** file if your system has sufficient VRAM to run it. Create a copy of the folder and rename it to something sensible, i.e. **yolov4-obj-custom.cfg**. In this file you want to change the following parameters to the following:

Under "# Training"

- batch=64
- subdivisions=32
- width=416
- height=416

Further down
- max_batches = 2000 * Number of Classes in your dataset (i.e 12 classes means max batches = 24000)
- steps = 0.8 * max_batches, 0.9 * max_batches

Then Ctrl+F and search for **"[yolo]"**.

Above each "[yolo]", change the number of filters to (number of classes + 5) * 3. So for 12 classes, (12 + 5) * 3 = 51 Filters.

Then below each "[yolo]", change the number of classes to the number of classes in your dataset.

Download either of the following weight files for [YOLOv4](https://drive.google.com/open?id=1JKF-bdIklxOOVy-2Cr5qdvjgGpmGfcbp) or [YOLOv4-tiny](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.conv.29) depending on what you will use to train your dataset.

Now we will run a python script to generate our desired "train.txt" file with the image names. Please check the ___ .py file.

To train the dataset after all the configuration, open an Anaconda command prompt and enter the following command:

darknet.exe detector train data/obj.data cfg/yolov4-tiny-obj.cfg yolov4-tiny.conv.29

This will begin the training process. Once this has been completed, navigate to the "Backup" folder in command line and run the following command:

darknet.exe detector map data/obj.data cfg/yolov4-tiny-obj.cfg backup/***name of weights file***

This will provide an mAP score of the weight file, take the file with the highest mAP. The last weights file is not necessarily always the greatest if there was a lot of overfitting during training.

Once the model is created, we can run tests to gauge the performance. A few examples of how my model performed on standard images can be seen here. From this point on, we use computer vision to extract important information about the objects being detected.

## Computer Vision

In this section, we use computer vision to extract key information from the objects detected, such as determining the state of detected traffic lights.

This project was carried out using Microsoft Visual Studio Code, where the python environment can be easily changed using the Interpreter. Ensure the python environment is set to the same environment which was used when OpenCV with CUDA capabilities was installed. This will provide optimal performance when using OpenCV's "DNN" module.

We also import a library called "CVZone". This helps to achieve easy overlaying of images, which we use to display instructions to the user dependent on the state of the objects that are detected. This allows us to use ".png" images to display messages rather than using the OpenCV text function. 

In the main python script, we convert the input video stream from RGB to HSV colour format. This allows us to specify a combination of HSV ranges to declare a colour. An example of specifying the HSV colour ranges of a red traffic light can be seen in the image below. This was achieved with the ___ python script. This process was performed on traffic lights and speed limit signs, for Red/Amber/Green traffic light states, and then Red/Blue for maximum/minimum speed limit states.

In the main python script, we open the "obj.names" file to store the class names of our objects in memory.
We also read the deep learning network with the weights file we created previously and the corresponding configuration file in the following line:

net = cv.dnn.readNet(weights, cfg)

We then specify the directory and the input video from which we want to run object detection on. The python script will split the video into individual frames and run detection on each frame before moving onto the next frame, giving us video playback with the detected objects shown.

We use OpenCV's "model.detect" to obtain the detected class along with its confidence score and box parameters which we can use to manually draw a rectangle around the deteted object. This is used in the following line:

classes, scores, boxes = model.detect(b, Conf_threshold, NMS_threshold)

We then get into the specifics of what should be done should a traffic light or a speed limit sign is detected. 

Using the HSV colour information we obtained with the ___ script, we can use OpenCV's perspectiveTransform and warpPerspective to obtain a bird's eye view of the object. We then dilate the object before using the "countNonZero" function to count the number of pixels within the specified colour ranges. The CVZone library is then used to overlay messages based on the state of the object.

