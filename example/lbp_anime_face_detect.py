'''Module
pip install opencv-python
'''
import cv2
import sys


def lbp_anime_face_detect(file_name):
    img = cv2.imread(file_name)  # 读取图片
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 图片灰度化
    img_gray = cv2.equalizeHist(img_gray)  # 直方图均衡化
    face_cascade = cv2.CascadeClassifier('../model/lbp_anime_face_detect.xml')  # 加载级联分类器
    faces = face_cascade.detectMultiScale(img_gray)  # 多尺度检测
    for x, y, w, h in faces:  # 遍历所有检测到的动漫脸
        img = cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 255), 5)  # 绘制矩形框
    cv2.imwrite(f'../result/lbp_anime_face_detect_{file_name[-5]}.jpg', img)  # 保存检测结果


if __name__ == '__main__':
    lbp_anime_face_detect(sys.argv[1])
