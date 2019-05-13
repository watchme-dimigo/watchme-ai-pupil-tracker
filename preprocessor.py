import cv2
from PIL import Image

def get_threshold(img):
    lightest = 0
    darkest = 255
    im = Image.fromarray(img)
    pixels = im.load()
    # print(im.size)
    for y in range(im.size[0]):
        for x in range(im.size[1]):
            color = pixels[y, x]
            if color > lightest:
                lightest = color
            elif color < darkest:
                darkest = color
    return (lightest + darkest) / 2

def apply_threshold(img):
    threshold = get_threshold(img)
    _, img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    return img
