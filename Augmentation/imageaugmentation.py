import imgaug.augmenters as iaa
import cv2
import glob
# final traffic dataset
# Apply grayscale,
# zoom in
# zoom out
# slight rotation (left)
# slight rotation (right)
# weather effects
# change brightness (dark)
# change brightness (bright)
# add blur to the images

# 1. Load Images
images = []
images_path = glob.glob("20mphspeedlimit_grayscale/*.jpg")
for img_path in images_path:
    img = cv2.imread(img_path)
    images.append(img)
#img = cv2.imread("20mphspeedlimit/20mph1.jpg")

# 2. Image Augmentation
augmentation = iaa.Sequential([
    # Add Grayscale
iaa.Grayscale(alpha=(0.0, 1.0))
# Add Fog
#iaa.Fog()
# Add Snow
# iaa.Snowflakes(flake_size=(0.7, 0.95), speed=(0.001, 0.03))
#  Add Rain
#iaa.Rain(drop_size=(0.10, 0.20)),
# Add perspective Transform
#iaa.PerspectiveTransform(scale=(0.01, 0.15))
# Rotate images
#iaa.Affine(rotate=(-45, 45))
# zoom 50% - 150% of initial size
#iaa.Affine(scale=(0.5, 1.5))
])
#iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0
#iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
#iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes between 2 and 7
#iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
augmented_images = augmentation(images=images)

# 3. Show Images
count = 1
for img in augmented_images:
    cv2.imshow("Image", img)
    cv2.imwrite('./20mphspeedlimit_grayscale/20mph' + str(count) + 'GRAYSCALE' + str(count) + '.jpg', img)
    count+=1
    cv2.waitKey(0)
