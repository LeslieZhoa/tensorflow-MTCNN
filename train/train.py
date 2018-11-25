
# coding: utf-8

# In[5]:


from model import P_Net,R_Net,O_Net
import argparse
import os
import sys
import config as FLAGS
from train_model import train
net_factorys=[P_Net,R_Net,O_Net]


# In[ ]:


def main(args):
    size=args.input_size
    base_dir=os.path.join('../data/',str(size))
    
    if size==12:
        net='PNet'
        net_factory=net_factorys[0]
        end_epoch=FLAGS.end_epoch[0]
    elif size==24:
        net='RNet'
        net_factory=net_factorys[1]
        end_epoch=FLAGS.end_epoch[1]
    elif size==48:
        net='ONet'
        net_factory=net_factorys[2]
        end_epoch=FLAGS.end_epoch[2]
    model_path=os.path.join('../model/',net)
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    prefix=os.path.join(model_path,net)
    display=FLAGS.display
    lr=FLAGS.lr
    train(net_factory,prefix,end_epoch,base_dir,display,lr)


# In[ ]:


def parse_arguments(argv):

    parser = argparse.ArgumentParser()

    parser.add_argument('input_size', type=int,
                        help='The input size for specific net')
    
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

