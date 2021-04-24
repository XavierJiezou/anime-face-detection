from example.lbp_anime_face_detect import lbp_anime_face_detect
# from example.mlp_anime_face_detect import mlp_anime_face_detect
from example.hog_anime_face_detect import hog_anime_face_detect
from example.ssd_anime_face_detect import ssd_anime_face_detect

import sys
import cv2


def image_detect(image_path, method):
    if method == 'lbp':
        img = lbp_anime_face_detect(image_path, './model/lbp_anime_face_detect.xml')
    # elif method=='mlp':
    #     img = mlp_anime_face_detect(image_path)
    elif method == 'hog':
        img = hog_anime_face_detect(image_path, './model/hog_anime_face_detect.svm')
    elif method == 'ssd':
        img = ssd_anime_face_detect(image_path, './model/ssd_anime_face_detect.pth')
    else:
        print('Input of Detection Method Error')
        sys.exit(0)
    return img


if __name__ == '__main__':
    cv2.imwrite('out.jpg', image_detect(sys.argv[1], sys.argv[2]))
