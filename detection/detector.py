
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np


# In[2]:


class Detector:
    '''识别多组图片'''
    def __init__(self,net_factory,data_size,batch_size,model_path):
        graph=tf.Graph()
        with graph.as_default():
            self.image_op=tf.placeholder(tf.float32,[None,data_size,data_size,3])
            self.cls_prob, self.bbox_pred, self.landmark_pred = net_factory(self.image_op, training=False)
            self.sess = tf.Session()
            #重载模型
            saver=tf.train.Saver()
            model_file=tf.train.latest_checkpoint(model_path)
            saver.restore(self.sess,model_file)
        self.data_size=data_size
        self.batch_size=batch_size
    def predict(self,databatch):
        scores=[]
        batch_size=self.batch_size
        minibatch=[]
        cur=0
        #所有数据总数
        n=databatch.shape[0]
        #将数据整理成固定batch
        while cur<n:
            minibatch.append(databatch[cur:min(cur+batch_size,n),:,:,:])
            cur+=batch_size
        cls_prob_list=[]
        bbox_pred_list=[]
        landmark_pred_list=[]
        for idx,data in enumerate(minibatch):
            m=data.shape[0]
            real_size=self.batch_size
            #最后一组数据不够一个batch的处理
            if m<batch_size:
                keep_inds=np.arange(m)
                gap=self.batch_size-m
                while gap>=len(keep_inds):
                    gap-=len(keep_inds)
                    keep_inds=np.concatenate((keep_inds,keep_inds))
                if gap!=0:
                    keep_inds=np.concatenate((keep_inds,keep_inds[:gap]))
                data=data[keep_inds]
                real_size=m
            cls_prob,bbox_pred,landmark_pred=self.sess.run([self.cls_prob, self.bbox_pred,self.landmark_pred],
                                                          feed_dict={self.image_op: data})
            
            cls_prob_list.append(cls_prob[:real_size])
            bbox_pred_list.append(bbox_pred[:real_size])
            landmark_pred_list.append(landmark_pred[:real_size])
        
        return np.concatenate(cls_prob_list, axis=0), np.concatenate(bbox_pred_list, axis=0), np.concatenate(landmark_pred_list, axis=0)

