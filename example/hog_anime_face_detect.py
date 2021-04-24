import cv2
import dlib
import sys


def hog_anime_face_detect(image_path, model_path):
    img = cv2.imread(image_path) if type(image_path)==str else image_path # 读取图片
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 图片灰度化
    img_gray = cv2.equalizeHist(img_gray)  # 直方图均衡化
    face_detector = dlib.simple_object_detector(model_path) # 加载检测器
    faces = face_detector(img_gray)
    for face in faces: # 遍历所有检测到的动漫脸
        left = face.left()
        top = face.top()
        right = face.right()
        bottom = face.bottom()
        cv2.rectangle(img, (left, top), (right, bottom), (255, 0, 255), 5) # 绘制矩形框
    # cv2.imwrite(f'../result/hog_anime_face_detect_{image_path[-5]}.jpg', img)  # 保存检测结果
    return img


if __name__ == '__main__':
    hog_anime_face_detect(sys.argv[1], '../model/hog_anime_face_detect.svm')
