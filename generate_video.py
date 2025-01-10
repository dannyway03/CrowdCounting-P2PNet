import os

import cv2


def images_to_video(images_folder, output_video_path, fps=30):
    # 获取文件夹中所有图片的路径
    images = [os.path.join(images_folder, f) for f in os.listdir(images_folder) if f.endswith('.jpg')]
    # 读取第一张图片，获取视频的尺寸
    image = cv2.imread(images[0])
    video_size = (image.shape[1], image.shape[0])
    # 创建视频对象，指定编码格式为mp4v，帧率为30，尺寸为video_size
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video_path, fourcc, fps, video_size)
    # 遍历所有图片，将图片写入视频
    for image in images:
        image = cv2.imread(image)
        video.write(image)
    # 释放视频对象
    video.release()


# 调用函数，将./images文件夹中的图片合成./output.mp4视频
images_to_video('/mnt/d/MyDocs/Datasets/mall_dataset/frames', './origin.mp4')
print('生成成功')

# import cv2
# import glob
#
# # 设置视频编解码器和帧率
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# fps = 30.0
#
# img_dir = '/mnt/d/MyDocs/Datasets/mall_dataset/frames'
#
# # 获取文件夹下的 JPG 图像
# img_files = glob.glob(f'{img_dir}/*.jpg')
#
# # 创建视频写入器对象
# width, height = cv2.imread(img_files[0]).shape[:2]
# video_out = cv2.VideoWriter('origin.mp4', fourcc, fps, (height, width))
#
# # 将 JPG 图像写入视频
# for img_file in img_files:
#     img = cv2.imread(img_file)
#     video_out.write(img)
#
# # 释放视频写入器对象和关闭视频文件
# video_out.release()
# cv2.destroyAllWindows()
# print('生成成功')
