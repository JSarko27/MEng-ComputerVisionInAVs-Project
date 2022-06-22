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
images_path = glob.glob("TrafficLights_fog/*.jpg")
for img_path in images_path:
    img = cv2.imread(img_path)
    images.append(img)


# 2. Image Augmentation
augmentation = iaa.Sequential([
# Add Fog
iaa.Fog()
])

augmented_images = augmentation(images=images)

# 3. Show Images
count = 1
for img in augmented_images:
    cv2.imshow("Image", img)
    cv2.imwrite('./TrafficLights/trafficlight' + str(count) + 'FOG' + str(count) + '.jpg', img)
    count+=1
    cv2.waitKey(0)
