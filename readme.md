# Introduction
This repository is the summary of anime face detection methods based on Python.
# Install
There are four methods for anime face detection: `lbp`, `mlp`, `hog` and `ssd`.
- `lbp`: `pip install opencv-python`
- `mlp`: `pip install pillow opencv-python animeface`
- `hog`: `pip install opencv-python dlib`
- `ssd`: `pip install opencv-python numpy torch`
# Usage
For image detection:
```bash
python ./image_detect.py ./test/1.jpg lbp
```
For video detection:
```bash
python ./video_detect.py ./test/1.mp4 lbp
```
# Test Sample
![在这里插入图片描述](https://img-blog.csdnimg.cn/2021042219445425.jpg#pic_center)
# Test Device
- CPU：`12  Intel(R) Xeon(R) CPU E5-2603 v4 @ 1.70GHz`
- GPU：`8 NVIDIA GeForce GTX 1080 Ti`
# Anime Face Detection
## Anime Face Detection Based on LBP
### Repository
> [https://github.com/nagadomi/lbpcascade_animeface](https://github.com/nagadomi/lbpcascade_animeface)
### Environment
- Module: `pip install opencv-python`
- Model: [lbp_anime_face_detect.xml](https://cdn.jsdelivr.net/gh/XavierJiezou/anime-face-detection@master/model/lbp_anime_face_detect.xml)
### Example
[lbp_anime_face_detect.py](https://cdn.jsdelivr.net/gh/XavierJiezou/anime-face-detection@master/example/lbp_anime_face_detect.py)
```python
import cv2
import sys


def lbp_anime_face_detect(file_name):
    img = cv2.imread(file_name)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.equalizeHist(img_gray) 
    face_cascade = cv2.CascadeClassifier('../model/lbp_anime_face_detect.xml')
    faces = face_cascade.detectMultiScale(img_gray)
    for x, y, w, h in faces:
        img = cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 255), 5)
    cv2.imwrite(f'../result/lbp_anime_face_detect_{file_name[-5]}.jpg', img) 


if __name__ == '__main__':
    lbp_anime_face_detect(sys.argv[1])
```
### Result
|Total| Missing  | Error | Time|
|:--:|:--:|:--:|:--:|
| 13 | 1 | 1 | 1.20s |

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210422194611765.jpg#pic_center)

## Anime Face Detection Based on MLP
### Repository
> [https://github.com/nya3jp/python-animeface](https://github.com/nya3jp/python-animeface)
### Environment
- System Requirements: Only for `Linux` system
- Module: `pip install pillow opencv-python animeface`

---
Note: If `pip install animeface` reports an error, please download the compiled file [animeface-1.1.0-cp37-cp37m-manylinux1_x86_64.whl.whl](https://files.pythonhosted.org/packages/d0/d9/40e9fdff3f9fa9dac27e1d687fc1af72efe355d040e6def6519c21e5e10a/animeface-1.1.0-cp37-cp37m-manylinux1_x86_64.whl), and then use the following command to install: (It is mandatory that the **Python** version is **3.7**)
```bash
pip install animeface-1.1.0-cp37-cp37m-manylinux1_x86_64.whl
```
### Example
[mlp_anime_face_detect.py](https://cdn.jsdelivr.net/gh/XavierJiezou/anime-face-detection@master/example/mlp_anime_face_detect.py)
```python
import cv2
import animeface
from PIL import Image
import sys


def mlp_anime_face_detect(file_name):
    img = cv2.imread(file_name)  
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
    img_gray = cv2.equalizeHist(img_gray)  
    faces = animeface.detect(Image.fromarray(img_gray)) 
    for each in faces:  
        temp = each.face.pos
        x = temp.x
        y = temp.y
        w = temp.width
        h = temp.height
        img = cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 255), 5)  
    cv2.imwrite(f'../result/mlp_anime_face_detect_{file_name[-5]}.jpg', img)  


if __name__ == '__main__':
    mlp_anime_face_detect(sys.argv[1])
```
### Result
|Total| Missing | Error | Time|
|:--:|:--:|:--:|:--:|
| 17 | 0 | 4 | 28.28s |

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210422195835588.jpg#pic_center)

## Anime Face Detection Based on HOG
### Repository
> [https://github.com/marron-akanishi/AFD](https://github.com/marron-akanishi/AFD)
### Environment
- Module: `pip install opencv-python dlib`
- Model: [hog_anime_face_detect.svm](https://cdn.jsdelivr.net/gh/XavierJiezou/anime-face-detection@master/model/hog_anime_face_detect.svm)
---
Note：The `dlib` library needs to be compiled with the `C++` compiler after downloading, so you need to install `Visual Studio` and configure the `C++` compilation environment. If `C++` environment have been installed and configured, please ignore; If not, please see [this Article](https://blog.csdn.net/qq_42951560/article/details/115949166).
### Example
[hog_anime_face_detect.py](https://cdn.jsdelivr.net/gh/XavierJiezou/anime-face-detection@master/example/hog_anime_face_detect.py)
```python
import cv2
import dlib
import sys


def hog_anime_face_detect(file_name):
    img = cv2.imread(file_name) 
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    img_gray = cv2.equalizeHist(img_gray)  
    face_detector = dlib.simple_object_detector('../model/hog_anime_face_detect.svm') 
    faces = face_detector(img_gray)
    for face in faces: 
        left = face.left()
        top = face.top()
        right = face.right()
        bottom = face.bottom()
        cv2.rectangle(img, (left, top), (right, bottom), (255, 0, 255), 5) 
    cv2.imwrite(f'../result/hog_anime_face_detect_{file_name[-5]}.jpg', img)  


if __name__ == '__main__':
    hog_anime_face_detect(sys.argv[1])
```
### Result
|Total| Missing | Error | Time|
|:--:|:--:|:--:|:--:|
| 10 | 3 | 0 | 2.42s |

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210422200142818.jpg#pic_center)

## Anime Face Detection Based on SSD
### Repository
> [https://github.com/WynMew/AnimeFaceBoxes](https://github.com/WynMew/AnimeFaceBoxes)
### Environment
- Module: `pip install opencv-python numpy torch`
- Model: [ssd_anime_face_detect.pth](https://cdn.jsdelivr.net/gh/XavierJiezou/anime-face-detection@master/model/ssd_anime_face_detect.pth)
### Example
[ssd_anime_face_detect.py](https://cdn.jsdelivr.net/gh/XavierJiezou/anime-face-detection@master/example/ssd_anime_face_detect.py) (The code is too long to display, Please download and view)
### Result
|Total| Missing | Error | Time|
|:--:|:--:|:--:|:--:|
| 13 | 0 | 0 | 0.72s |

![在这里插入图片描述](https://img-blog.csdnimg.cn/2021042221243255.jpg#pic_center)
# Other Experiment
## Test Sample
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210422221239814.jpg#pic_center)
<center><font color=#CCCCCC>【Painter】ゆりりん【Pixiv ID】88049646 </font></center>

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210422221239888.jpg#pic_center)
<center><font color=#CCCCCC>【Painter】Nahaki【Pixiv ID】84678881 </font></center>

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210422221239701.jpg#pic_center)
<center><font color=#CCCCCC>【Painter】A.one【Pixiv ID】66996496 </font></center>

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210422221239690.jpg#pic_center)
<center><font color=#CCCCCC>【Painter】D.【Pixiv ID】68074033 </font></center>


## Test Result
![在这里插入图片描述](https://img-blog.csdnimg.cn/2021042300100255.jpg#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210423001001987.jpg#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/2021042300100136.jpg#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/202104230010028.jpg#pic_center)


# Analysis 
The anime face detection algorithm based on **MLP** is too slow to meet the requirements of practical applications. The other three algorithms behave differently on different sample image.
# Future Work
- Anime Face Detection Based on Faster-RCNN: https://github.com/qhgz2013/anime-face-detector/
- Anime Face Detection Based on CNN: [https://github.com/ShiqiYu/libfacedetection](https://github.com/ShiqiYu/libfacedetection)
- Other Related Work: [https://github.com/search?p=1&q=anime+face+detection&type=Repositories](https://github.com/search?p=1&q=anime%20face%20detection&type=Repositories)
# Cite
> [https://github.com/nagadomi/lbpcascade_animeface](https://github.com/nagadomi/lbpcascade_animeface)

> [https://github.com/nya3jp/python-animeface](https://github.com/nya3jp/python-animeface)

> [https://github.com/marron-akanishi/AFD](https://github.com/marron-akanishi/AFD)

> [https://github.com/WynMew/AnimeFaceBoxes](https://github.com/WynMew/AnimeFaceBoxes)

> [https://github.com/hiromu/AnimeFace](https://github.com/hiromu/AnimeFace)
