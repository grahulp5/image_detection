import albumentations as A
import cv2
import os
from xml.dom import minidom
import random
import copy
from draw_boxes import draw_boxes


# Your image path - an example image is added in
imagespath = '.Example'
# Directory where the augmentation WITHOUT bounding boxes are saved
output_dir = './Data_augmented/'
# Directory where the augmentation WITH bounding boxes are saved
output_dir_box = './Data_augmented_box/'
random.seed(7)
# This creates your directory if it does not exist yet
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_dir_box, exist_ok=True)


def readImage(filename):
    # OpenCV uses BGR channels
    img = cv2.imread(imagespath+filename)
    return img


# Extraction of bounding box annotations and labels of the images
def getCoordinates(filename):
    allbb = []
    xmldoc = minidom.parse(imagespath+filename)
    itemlist = xmldoc.getElementsByTagName('object')

    size = xmldoc.getElementsByTagName('size')[0]
    width = int((size.getElementsByTagName('width')[0]).firstChild.data)
    height = int((size.getElementsByTagName('height')[0]).firstChild.data)

    for item in itemlist:
        classid = (item.getElementsByTagName('name')[0]).firstChild.data
        xmin = ((item.getElementsByTagName('bndbox')[
            0]).getElementsByTagName('xmin')[0]).firstChild.data
        ymin = ((item.getElementsByTagName('bndbox')[
            0]).getElementsByTagName('ymin')[0]).firstChild.data
        xmax = ((item.getElementsByTagName('bndbox')[
            0]).getElementsByTagName('xmax')[0]).firstChild.data
        ymax = ((item.getElementsByTagName('bndbox')[
            0]).getElementsByTagName('ymax')[0]).firstChild.data

        xmin = int(xmin)
        ymin = int(ymin)
        xmax = int(xmax)
        ymax = int(ymax)

        b = [xmin, ymin, xmax, ymax, classid]
        allbb.append(b)
    return allbb


# Start of the data augmentation process and this reads the images and saves them in your directories.
def start():
    for filename in sorted(os.listdir(imagespath)):

        if filename.endswith(".jpg") or filename.endswith(".JPG"):
            title, ext = os.path.splitext(os.path.basename(filename))
            image = readImage(filename)

        if filename.endswith(".txt"):
            xmlTitle, txtExt = os.path.splitext(os.path.basename(filename))
            if xmlTitle == title:
                # bboxes = getCoordinates(filename)
                bboxes = readYolo(imagespath+xmlTitle+'.txt')
                for i in range(0, 11):
                    img = copy.deepcopy(image)
                    transform = getTransform(i)
                    try:
                        transformed = transform(image=img, bboxes=bboxes)
                        transformed_image = transformed['image']
                        transformed_bboxes = transformed['bboxes']
                        name = title + str(i) + '.jpg'

                        annot_image, box_areas = draw_boxes(transformed_image, transformed_bboxes, 'yolo')
                        annot_image = cv2.cvtColor(annot_image, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(output_dir_box + name, annot_image)

                        # transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR) # I think this is the reason why the color is different.
                        cv2.imwrite(output_dir + name, transformed_image)
                        # print(transformed_bboxes)
                        # writeVoc(transformed_bboxes, count, transformed_image)
                        writeYolo(transformed_bboxes, i, title)
                    except:
                        print("bounding box issues")
                        pass


# Reads bounding box coordinates and converted into a list
def readYolo(filename):
    coords = []
    with open(filename, 'r') as fname:
        for file1 in fname:
            x = file1.strip().split(' ')
            x.append(x[0])
            x.pop(0)
            x[0] = float(x[0])
            x[1] = float(x[1])
            x[2] = float(x[2])
            x[3] = float(x[3])
            coords.append(x)
    return coords

# Writing of bounding box coordinates in YOLO format to a txt file
def writeYolo(coords, count, name):

    with open(output_dir + name+str(count)+'.txt', "w") as f:
        for x in coords:
            f.write("%s %s %s %s %s \n" % (x[-1], x[0], x[1], x[2], x[3]))


# Different types of Data Augmentation (10 procedures)
def getTransform(loop):
    # This filps the image horizontally
    if loop == 0:
        transform = A.Compose([
            A.HorizontalFlip(p=1),
        ], bbox_params=A.BboxParams(format='yolo'))
        # This randomly changes the brightness and contrast. This can be adjusted by modifying the limit (Max = 1)
    elif loop == 1:
        transform = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.5, p=1),
        ], bbox_params=A.BboxParams(format='yolo'))
        # This randomly changes the brightness of the image on a per pixel basis (the higher the multiplier, the darker the image).
        # This helps in introducing variations to brightness.
    elif loop == 2:
        transform = A.Compose([
            A.MultiplicativeNoise(multiplier=0.5, p=1),
        ], bbox_params=A.BboxParams(format='yolo'))
        # This flips the image vertically.
    elif loop == 3:
        transform = A.Compose([
            A.VerticalFlip(p=1)
        ], bbox_params=A.BboxParams(format='yolo'))
        # This rotates the image by 90 degrees randomly in any direction.
    elif loop == 4:
        transform = A.Compose([
            A.RandomRotate90(p=1)
        ], bbox_params=A.BboxParams(format='yolo'))
        # This compresses the image to get it on a lower or file size.
        # It is of note that this compression only applies to JPEG images.
        # It is of note as well that this helps simulate the artifacts as a result of compression.
    elif loop == 5:
        transform = A.Compose([
            A.JpegCompression(quality_lower=0, quality_upper=1, p=1)
        ], bbox_params=A.BboxParams(format='yolo'))
        # This applies PCA, but on the pixel values based on RGB colors.
        # This introduces variations in color as a result of the linear combinations created.
    elif loop == 6:
        transform = A.Compose([
            A.FancyPCA(alpha=0.6, p=1)
        ], bbox_params=A.BboxParams(format='yolo'))
        # This applies blurring to the image, and can help simulate out-of-focus images.
    elif loop == 7:
        transform = A.Compose([
            A.Blur(blur_limit=30, p=1)
        ], bbox_params=A.BboxParams(format='yolo'))
        # This applies Gaussian Noise, which adds random variations that mimic noise in real-world images.
    elif loop == 8:
        transform = A.Compose([
            A.GaussNoise(var_limit=(0, 100.0), p=1)
        ], bbox_params=A.BboxParams(format='yolo'))
        # This introduces RGB shifts, which can help understand color distortions present.
        # p = 1 means that RGB shift will always be applied
    elif loop == 9:
        transform = A.Compose([
            A.RGBShift(p=1)
        ], bbox_params=A.BboxParams(format='yolo'))
        # p = 0 means that no RGB shift will be applied.
    elif loop == 10:
        transform = A.Compose([
            A.RGBShift(p=0)
        ], bbox_params=A.BboxParams(format='yolo'))

    return transform

start()