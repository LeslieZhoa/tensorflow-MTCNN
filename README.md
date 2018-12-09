# tensorflow-MTCNN
人脸检测MTCNN算法，采用tensorflow框架编写，从理解到训练，中文注释完全，含测试和训练，支持摄像头，代码参考[AITTSMD](https://github.com/AITTSMD/MTCNN-Tensorflow)，做了相应删减和优化。
## 模型理解
[MTCNN](https://kpzhang93.github.io/MTCNN_face_detection_alignment/index.html)是目前比较流行的人脸检测方法，通过人脸检测可以进行更精准的人脸识别。模型主要通过PNet，RNet，ONet三个网络级联，一步一步精调来对人脸进行更准确的检测。论文中的模型图如下：<br>
![](https://github.com/LeslieZhoa/tensorflow-MTCNN/blob/master/output/model1.png)<br>
![](https://github.com/LeslieZhoa/tensorflow-MTCNN/blob/master/output/model2.png)<br>
接下来我会从我在训练中的理解来解释MTCNN模型都干了什么。<br><br>
三个模型要按顺序训练，首先是PNet,然后RNet,最后ONet。<br><br>
PNet:<br><br>
PNet是全卷积网络，主要为了应对不同输入尺度，层数很浅，主要作用是尽可能多的把人脸框都选进来，宁愿错误拿来好多个，也不丢掉一个。训练数据由四部分组成：pos,part,neg,landmark，比例为1：1：3：1。数据是怎么来的呢？<br><br>
pos,part,neg是随机和人脸的数据裁剪得到的，裁剪图片与人脸框最大的iou值大于0.65的为pos图像，大于0.4的为part图像，小于0.3的为neg图像，landmark截取的是带有关键点的图像。其中pos,part的label含有它们的类别1,-1还有人脸框相对于图像左上角的偏移量，偏移量除以图像大小做了归一化;neg的label只含有类别0;landmark的label含有类别-2和5个关键点的坐标偏移也是进行了归一化的。<br><br>
这四种图像都resize成12x12作为PNet的输入，通过PNet得到了是否有人脸的概率[batch,2]，人脸框的偏移量[batch,4]，关键点的偏移量[batch,10]。四种不同的数据该怎么训练呢？<br><br>
对于是否存在人脸的类别损失只通过neg和pos数据来对参数进行更新，具体办法是通过label中的类别值做了一个遮罩来划分数据，只计算neg和pos的损失，不计算其他数据的损失;人脸框的损失只计算pos和part数据的;关键点的损失只计算landmark的。论文中有个小技巧就是只通过前70%的数据进行更新参数，说是模型准确率会有提升，在代码中也都有体现，具体实现可以参考代码。<br><br>
RNet,ONet:<br><br>
RNet和ONet都差不多都是精修人脸框，放在一起解释。RNet的landmark数据和PNet一样，是对带有关键点的图像截取得到，但要resize成24x24的作为输入。<br><br>
pos,part,neg的数据是通过PNet得到的。这里就可以理解为什么PNet的输入要是四种数据大小是12了，为了速度，也为了RNet的输入。一张图片输入到PNet中会得到[1,h,w,2],[1,h,w,4],[1,h,w,10]的label预测值，这有点像yolo的思想，如果不理解yolo的可以参考[我的yolo介绍](https://github.com/LeslieZhoa/tensorflow-YOLO1)。<br><br>
把一张图片像网格一样划分，每一个网格都预测它的人脸框，划分的图片包含的人脸有多有少，所以划分了neg，pos，part三种数据，landmark只是起辅助作用。图片还要以一定值来缩小尺度做成图像金字塔目的是获取更多可能的人脸框，人脸框中有人的概率大于一定阈值才保留，还要进行一定阈值的非极大值抑制，将太过重合的人脸框除掉，将PNet预测的人脸框于原图上截取，与真实人脸框的最大iou值来划分neg，pos,part数据，并resize成24作为RNet的输入。<br><br>
RNet，ONet的损失函数和PNet相同，不同的是三种损失所占的比例不同。<br>ONet的输入是图片通过PNet金字塔得到的裁剪框再经过RNet的裁剪框裁剪的图片划分neg,pos,part三种数据resize成48作为输入，landmark与RNet相同只不过resize成48大小的了。<br><br>
## 代码介绍
### 环境说明
ubuntu16.04<br>
python3.6.5<br>
tensorflow1.8.0<br>
opencv3.4.3<br>
pip install tqdm为了显示进度条<br>
### 代码介绍
data下放置训练所用的原始数据和划分数据，生成的tfrecord等<br><br>
detection下的fcn_detector.py主要用于PNet的单张图片识别，detector.py用于RNet和ONet的一张图片通过PNet截取的多个人脸框的批次识别，MtcnnDetector.py为识别人脸和生成RNet，ONet输入数据<br><br>
graph里放置的是训练过程中生成的graph文件<br><br>
output里放置识别图像或视频完成后存储放置的路径<br><br>
picture里是要测试的图像放置路径<br><br>
preprocess里是预处理数据程序，BBox_utils.py和utils.py，loader.py是一些辅助程序，gen_12net_data.py是生成PNet的pos，neg,part的程序，gen_landmark_aug.py是生成landmark数据的程序，gen_imglist_pnet.py是pnet的四种数据组合一起，gen_hard_example.py是生成rnet,onet的三种数据程序，gen_tfrecords.py是生成tfrecord文件的程序<br><br>
train中的config是一些参数设定，大都文件夹我都直接写死了，所以里面参数能改的很少，model.py是模型,train.py是训练，train_model.py针对不同网络训练<br><br>
test.py是测试代码<br>
### 下载数据
将[WIDERFace](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/)的训练数据下载解压，将里面的WIDER_train文件夹放置到data下，将[Deep Convolutional Network Cascade for Facial Point Detection的训练集](http://mmlab.ie.cuhk.edu.hk/archive/CNN_FacePoint.htm)解压，将里面的lfw_5590和net_7876文件夹放置到data下。model文件夹下已存储好我训练的权值文件了。<br>

### 运行
训练：<br><br>
将目录cd到preprocess上，<br>
python gen_12net_data.py生成三种pnet数据，<br>
python gen_landmark_aug.py 12 生成pnet的landmark数据，<br>
python gen_imglist_pnet.py整理到一起，<br>
python gen_tfrecords.py 12生成tfrecords文件<br>
将目录cd到train上python train.py 12 训练pnet<br><br>
将目录cd到preprocess上，<br>
python gen_hard_example.py 12 生成三种rnet数据，<br>
python gen_landmark_aug.py 24 生成rnet的landmark数据,<br>
python gen_tfrecords.py 24生成tfrecords文件<br>
将目录cd到train上python train.py 24 训练rnet<br><br>
将目录cd到preprocess上，<br>
python gen_hard_example.py 24 生成三种onet数据，<br>
python gen_landmark_aug.py 48 生成onet的landmark数据,<br>
python gen_tfrecords.py 48生成tfrecords文件<br>
将目录cd到train上python train.py 48 训练onet<br><br>
测试:<br><br>
python test.py<br>
### 一些建议
生成hard_example时间非常长需要三到四小时，所以如果你想从头训练请耐心等待，如果代码或理解有什么问题，欢迎批评指正。<br>
### 结果展示
测试图片来源百度图片，测试结果如下：<br>
![](https://github.com/LeslieZhoa/tensorflow-MTCNN/blob/master/output/2007_000346.jpg)<br>
![](https://github.com/LeslieZhoa/tensorflow-MTCNN/blob/master/output/test.jpg)<br>
![](https://github.com/LeslieZhoa/tensorflow-MTCNN/blob/master/output/w475_h331_9a5169d0369e4e1496d1cdfabb1ded85.jpg)<br>
