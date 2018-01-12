from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib
import matplotlib.pyplot as plt
import pylab
import os, sys
import cv2

def load_data_coco(file_path, skip = None, selected = None):
    # val
    annFile = os.path.join(file_path, 'captions.json')

    coco_caps = COCO(annFile)

    imgIds = coco_caps.getImgIds() if selected is None else selected
    
    if skip is not None and skip is not 0:
        new_img_ids = []
        for i in range(len(imgIds)):
            if i % skip == 0:
                new_img_ids.append(imgIds[i])
        imgIds = new_img_ids
    
    img_dicts = coco_caps.loadImgs(imgIds)

    N = len(img_dicts)
    imgs, anns = np.zeros(shape=[N, 224, 224, 3], dtype=np.float32), []

    image_loader = ImageLoader()
    for i in range(N):
        if i % 1000 == 0:
            print('loading images : {}/{}'.format(i, N))
        
        img_dict = img_dicts[i]
        img = image_loader.load_img(os.path.join(file_path, 'image', img_dict['file_name']))
        #img = io.imread(os.path.join(file_path, 'image', img_dict['file_name']))
        imgs[i] = np.array(img)
        assert img.shape[0] + img.shape[1] + img.shape[2] == 3 + 224 + 224
        annIds = coco_caps.getAnnIds(imgIds=img_dict['id'])
        anns_tem = coco_caps.loadAnns(annIds)
        anns_tem = [process(ann_tem['caption']) for ann_tem in anns_tem]
        anns.append(anns_tem)

    anns = np.array(anns)
    return imgs, anns
    
def process(cap):
    while (cap[-1] == ' ') : cap = cap[: -1]
    if (cap[-1] != '.') : 
        cap = cap + ' .'
    else:
        cap = cap.replace('.', ' .')
    return cap

def show_img(img):
    pylab.rcParams['figure.figsize'] = (8., 10.)
    plt.axis('off')
    plt.imshow(img.astype(np.uint8))
    plt.show()

def show_caps(caps):
    for cap in caps:
        print(cap['caption'])

class ImageLoader(object):
    def __init__(self, mean_file = os.path.join('pycocotools', 'ilsvrc_2012_mean.npy')):
        self.bgr = True
        self.scale_shape = np.array([224, 224], np.int32)
        self.crop_shape = np.array([224, 224], np.int32)
        self.mean = np.load(mean_file).mean(1).mean(1)
        self.mean[[0, 1, 2]] = self.mean[[2, 1, 0]]
        
    def load_img(self, img_file):    
        """ Load and preprocess an image. """  
        img = cv2.imread(img_file)
        
        '''
        if self.bgr:
            temp = img.swapaxes(0, 2)
            temp = temp[::-1]
            img = temp.swapaxes(0, 2)
        '''

        img = cv2.resize(img, (self.scale_shape[0], self.scale_shape[1]))
        '''
        offset = (self.scale_shape - self.crop_shape) / 2
        offset = offset.astype(np.int32)
        img = img[offset[0]:offset[0]+self.crop_shape[0], offset[1]:offset[1]+self.crop_shape[1], :]
        '''
        img = img - self.mean
        return img

    def load_imgs(self, img_files):
        """ Load and preprocess a list of images. """
        imgs = []
        for img_file in img_files:
            imgs.append(self.load_img(img_file))
        imgs = np.array(imgs, np.float32)
        return imgs

    def bgrgbgr(self, img):
        temp = img.swapaxes(0, 2)
        temp = temp[::-1]
        img = temp.swapaxes(0, 2)
        return img

    def get_origin_image(self, img):
        img = img + self.mean
        #img = self.bgrgbgr(img)
        return img