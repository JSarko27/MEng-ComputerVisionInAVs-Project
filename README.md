# MEng-AV-Project
Final Year individual project on how computer vision can be implemented in autonomous vehicles with object detection
Object detection was achieved by labelling my own dataset and training on YOLOv4-tiny network

# Thesis for Project
For more details regarding the process and findings from this project, see the **thesis.pdf** file.

# Installing YOLOv4
To install YOLOv4 locally, the following software prerequisites are required:

Microsoft Visual Studio Community 2019

Git 2.35

Microsoft Powershell

CUDA Computing Toolkit 11.6

cuDNN 8.3.2

CMake 3.22.2

OpenCV 4.5.2

OpenCV-contrib 4.5.2

[MinGW](https://sourceforge.net/projects/mingw/)

The first step to be taken is to ensure that your system has the required software development environment required to compile programs later during the installation phase. This is achieved by installing the [MinGW](https://sourceforge.net/projects/mingw/) base compiler first.

Once the software is installed, to install the necessary compiler, only **mingw32-base** & **mingw32-gcc-g++** were selected before proceeding with the install, however there is no harm in selecting more packages, as long as the aforementioned packages are selected.

The next steps can be found on the original [YOLOv4](https://github.com/AlexeyAB/darknet#how-to-compile-on-windows-using-vcpkg) repository. When installing the packages, ensure environment variables are designated to the correct directories. Ensure the option "**-EnableOPENCV_CUDA**" is typed in powershell before running the vcpkg install in command line, this will ensure that OpenCV's GPU capabilities can be fully utilised.

Here are some good step by step video tutorials to install YOLO and OpenCV if you run into any issues.

[YOLO Install](https://www.youtube.com/watch?v=WK_2bpWj35A)

[OpenCV with CUDA](https://www.youtube.com/watch?v=HsuKxjQhFU0&t=1148s)

# Creating a Dataset

