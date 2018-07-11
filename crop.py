import cv2

def crop(path, posx1, posy1, posx2, posy2):
    """ Crop Function take path, x1, y1, x2, y2 then give back cropped photo """

    data = cv2.imread(path)

    cropped = data[posy1: posy1 + abs(posy2 - posy1), posx1: posx1 + abs(posx2 - posx1)]

    return cropped

