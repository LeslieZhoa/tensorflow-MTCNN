
# coding: utf-8

# In[ ]:


#迭代次数
end_epoch=[30,22,22]
#经过多少batch显示数据
display=100
#初始学习率
lr=0.001

batch_size=384
#学习率减少的迭代次数
LR_EPOCH=[6,14,20]
#最小脸大小设定
min_face=20

#生成hard_example的batch
batches=[2048,256,16]
#pent对图像缩小倍数
stride=2
#三个网络的阈值
thresh=[0.6,0.7,0.7]
#最后测试选择的网络
test_mode='ONet'
#选用图片还是摄像头,1是图像，2是摄像头
input_mode='2'
#测试图片放置位置
test_dir='picture/'
#测试输出位置
out_path='output/'

