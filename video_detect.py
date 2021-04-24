import sys
import cv2
import time
from tqdm import tqdm
from image_detect import image_detect
from concurrent.futures import ThreadPoolExecutor


def show(num, _sum,  runTime):
    barLen = 20  # 进度条的长度
    perFin = num/_sum
    numFin = round(barLen*perFin)
    numNon = barLen-numFin
    leftTime = (1-perFin)*(runTime/perFin)
    print(
        f"{num:0>{len(str(_sum))}}/{_sum}",
        f"|{'█'*numFin}{' '*numNon}|",
        f"任务进度: {perFin*100:.2f}%",
        f"已用时间: {runTime/60:.2f}m",
        f"剩余时间: {leftTime/60:.2f}m",
        end='\r'
    )
    if num == _sum:
        print()


def video_detect(video_path, method):
    video = cv2.VideoCapture(video_path)  # 加载视频
    fps = video.get(cv2.CAP_PROP_FPS)  # 帧率
    w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))  # 宽
    h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 高
    fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')  # 指定视频编码方式
    videoWriter = cv2.VideoWriter(f'out.mp4', 0x7634706d, fps, (w, h))  # 创建视频写对象
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))  # 视频总帧数
    imgae_list = []
    for i in tqdm(range(frame_count)):  # 帧数遍历
        success, img = video.read()  # 读取视频帧
        imgae_list.append(img)
        # img = image_detect(img, method)  # 帧检测
        # videoWriter.write(img)  # 视频对象写入
    tp = ThreadPoolExecutor(32)
    t1 = time.time()
    num = 0
    for result in tp.map(image_detect, imgae_list, [method]*frame_count):
        videoWriter.write(result)
        num+=1
        t2 = time.time()
        show(num, frame_count, t2-t1)


if __name__ == "__main__":
    video_detect(sys.argv[1], sys.argv[2])
