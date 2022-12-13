# import required libraries
import os
import cv2
import numpy as np

folder_dir = "Python Scripts\images"
list_images = []
for images in os.listdir(folder_dir):
    if images.endswith("jpg"):
        list_images.append(images)

print(list_images)

# read input image
for i in list_images:
    img = cv2.imread(os.path.join("Python Scripts\images", i))


    scale_percent = 50

    #calculate the 50 percent of original dimensions
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)

    # dsize
    dsize = (width, height)

    # resize image
    output = cv2.resize(img, dsize)


    # Convert BGR to HSV
    hsv = cv2.cvtColor(output, cv2.COLOR_BGR2HSV)

    # define range of blue color in HSV
    lower_yellow = np.array([0,0,70])
    upper_yellow = np.array([173,190,155])

    # Create a mask. Threshold the HSV image to get only yellow colors
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Bitwise-AND mask and original image
    result = cv2.bitwise_and(output,output, mask= mask)

    cv2.imwrite(os.path.join("Python Scripts\masked_images", f"masked_image_{i}"), mask)

    # display the mask and masked image
    #cv2.imshow('Mask',mask)
    cv2.waitKey(0)
    #cv2.imshow('Masked Image',result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()