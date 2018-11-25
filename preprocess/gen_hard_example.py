
# coding: utf-8

# In[1]:


import sys
from utils import *
import numpy as np
import argparse
import os
import pickle
import cv2
from tqdm import tqdm
from loader import TestLoader
sys.path.append('../')
from train.model import P_Net,R_Net,O_Net
import train.config as config
from detection.detector import Detector
from detection.fcn_detector import FcnDetector
from detection.MtcnnDetector import MtcnnDetector


# In[3]:


def main(args):
    '''通过PNet或RNet生成下一个网络的输入'''
    size=args.input_size
    batch_size=config.batches
    min_face_size=config.min_face
    stride=config.stride
    thresh=config.thresh
    #模型地址
    model_path=['../model/PNet/','../model/RNet/','../model/ONet']
    if size==12:
        net='PNet'
        save_size=24
    elif size==24:
        net='RNet'
        save_size=48
    #图片数据地址
    base_dir='../data/WIDER_train/'
    #处理后的图片存放地址
    data_dir='../data/%d'%(save_size)
    neg_dir=os.path.join(data_dir,'negative')
    pos_dir=os.path.join(data_dir,'positive')
    part_dir=os.path.join(data_dir,'part')
    for dir_path in [neg_dir,pos_dir,part_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    detectors=[None,None,None]
    PNet=FcnDetector(P_Net,model_path[0])
    detectors[0]=PNet
    if net=='RNet':
        RNet=Detector(R_Net,24,batch_size[1],model_path[1])
        detectors[1]=RNet
    basedir='../data/'
    filename='../data/wider_face_train_bbx_gt.txt'
    #读取文件的image和box对应函数在utils中
    data=read_annotation(base_dir,filename)
    mtcnn_detector=MtcnnDetector(detectors,min_face_size=min_face_size,
                                stride=stride,threshold=thresh)
    save_path=data_dir
    save_file=os.path.join(save_path,'detections.pkl')
    if not os.path.exists(save_file):
        #将data制作成迭代器
        print('载入数据')
        test_data=TestLoader(data['images'])
        detectors,_=mtcnn_detector.detect_face(test_data)
        print('完成识别')

        with open(save_file,'wb') as f:
            pickle.dump(detectors,f,1)
    print('开始生成图像')
    save_hard_example(save_size,data,neg_dir,pos_dir,part_dir,save_path)


# In[2]:


def save_hard_example(save_size, data,neg_dir,pos_dir,part_dir,save_path):
    '''将网络识别的box用来裁剪原图像作为下一个网络的输入'''

    im_idx_list = data['images']
    
    gt_boxes_list = data['bboxes']
    num_of_images = len(im_idx_list)


    
    # save files
    neg_label_file = "../data/%d/neg_%d.txt" % (save_size, save_size)
    neg_file = open(neg_label_file, 'w')

    pos_label_file = "../data/%d/pos_%d.txt" % (save_size, save_size)
    pos_file = open(pos_label_file, 'w')

    part_label_file = "../data/%d/part_%d.txt" % (save_size, save_size)
    part_file = open(part_label_file, 'w')
    #read detect result
    det_boxes = pickle.load(open(os.path.join(save_path, 'detections.pkl'), 'rb'))
    # print(len(det_boxes), num_of_images)
   
    assert len(det_boxes) == num_of_images, "弄错了"

    
    n_idx = 0
    p_idx = 0
    d_idx = 0
    image_done = 0
    
    for im_idx, dets, gts in tqdm(zip(im_idx_list, det_boxes, gt_boxes_list)):
        gts = np.array(gts, dtype=np.float32).reshape(-1, 4)
        image_done += 1

        if dets.shape[0] == 0:
            continue
        img = cv2.imread(im_idx)
        #转换成正方形
        dets = convert_to_square(dets)
        dets[:, 0:4] = np.round(dets[:, 0:4])
        neg_num = 0
        for box in dets:
            x_left, y_top, x_right, y_bottom, _ = box.astype(int)
            width = x_right - x_left + 1
            height = y_bottom - y_top + 1

            # 除去过小的
            if width < 20 or x_left < 0 or y_top < 0 or x_right > img.shape[1] - 1 or y_bottom > img.shape[0] - 1:
                continue

           
            Iou = IOU(box, gts)
            cropped_im = img[y_top:y_bottom + 1, x_left:x_right + 1, :]
            resized_im = cv2.resize(cropped_im, (save_size, save_size),
                                    interpolation=cv2.INTER_LINEAR)

            #划分种类           
            if np.max(Iou) < 0.3 and neg_num < 60:
                
                save_file = os.path.join(neg_dir, "%s.jpg" % n_idx)
                
                neg_file.write(save_file + ' 0\n')
                cv2.imwrite(save_file, resized_im)
                n_idx += 1
                neg_num += 1
            else:
               
                idx = np.argmax(Iou)
                assigned_gt = gts[idx]
                x1, y1, x2, y2 = assigned_gt

                #偏移量
                offset_x1 = (x1 - x_left) / float(width)
                offset_y1 = (y1 - y_top) / float(height)
                offset_x2 = (x2 - x_right) / float(width)
                offset_y2 = (y2 - y_bottom) / float(height)

                # pos和part
                if np.max(Iou) >= 0.65:
                    save_file = os.path.join(pos_dir, "%s.jpg" % p_idx)
                    pos_file.write(save_file + ' 1 %.2f %.2f %.2f %.2f\n' % (
                        offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    p_idx += 1

                elif np.max(Iou) >= 0.4:
                    save_file = os.path.join(part_dir, "%s.jpg" % d_idx)
                    part_file.write(save_file + ' -1 %.2f %.2f %.2f %.2f\n' % (
                        offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    d_idx += 1
    neg_file.close()
    part_file.close()
    pos_file.close()


# In[4]:


def parse_arguments(argv):

    parser = argparse.ArgumentParser()

    parser.add_argument('input_size', type=int,
                        help='The input size for specific net')
    
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

