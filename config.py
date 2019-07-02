"""
Author  : Xu fuyong
Time    : created by 2019/6/28 17:04

"""
import os

base_dir = './images'

img_dir = os.path.join(base_dir, 'VOC2007_text_detection/JPEGImages')
xml_dir = os.path.join(base_dir, 'VOC2007_text_detection/Annotations')

train_txt_file = os.path.join(base_dir, r'VOC2007_text_detection/ImageSets/Main/train.txt')
val_txt_file = os.path.join(base_dir, r'VOC2007_text_detection/ImageSets/Main/val.txt')

# 固定宽度
anchor_scale = 16
IOU_NEGATIVE = 0.3
IOU_POSITIVE = 0.7
IOU_SELECT = 0.7

RPN_POSITIVE_NUM = 150
RPN_TOTAL_NUM = 300

# https://github.com/fchollet/deep-learning-models/blob/master/imagenet_utils.py
IMAGE_MEAN = [123.68, 116.779, 103.939]

checkpoints_dir = './checkpoints'
# r是防止字符转义的 如果路径中出现'\t'的话 不加r的话\t就会被转义
outputs = r'./logs'


