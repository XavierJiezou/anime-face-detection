import cv2
import animeface
from PIL import Image
import sys


def mlp_anime_face_detect(image_path):
    img = cv2.imread(image_path) if type(image_path)==str else image_path  # 读取图片
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 图片灰度化
    img_gray = cv2.equalizeHist(img_gray)  # 直方图均衡化
    faces = animeface.detect(Image.fromarray(img_gray))  # 人脸检测
    for each in faces:  # 遍历所有检测到的动漫脸
        temp = each.face.pos
        x = temp.x
        y = temp.y
        w = temp.width
        h = temp.height
        img = cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 255), 5)  # 绘制矩形框
    # cv2.imwrite(f'../result/mlp_anime_face_detect_{image_path[-5]}.jpg', img)  # 保存检测结果
    return img


if __name__ == '__main__':
    mlp_anime_face_detect(sys.argv[1])
