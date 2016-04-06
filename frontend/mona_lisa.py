import numpy as np
import caffe
import cv2
from scipy.spatial import distance
from sklearn import svm
import cPickle as pickle

class VGGFeatureExtractor:
    def __init__(self, model_weights, model_def,
                mean=[104.0, 117.0, 123.0], 
                new_shape=(256, 256), crop=(244, 244)):
        self.net = caffe.Net(model_def, model_weights, caffe.TEST)
        self.net.blobs['data'].reshape(1, 3, 224, 224)
        self.transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
        self.transformer.set_transpose('data', (2,0,1))
        self.transformer.set_mean('data', np.array(mean))
        self.transformer.set_raw_scale('data', 255)
        self.transformer.set_channel_swap('data', (2,1,0))
        self.new_shape = new_shape
        self.crop = crop

    def reshape_image(self, image, new_dim=None, crop_dim=None):
        if new_dim is None:
            new_dim = self.new_shape
        if crop_dim is None:
            crop_dim = self.crop
        reshaped_img = cv2.resize(image, new_dim)
        crop_amt = (new_dim[0]-crop_dim[0])/2, (new_dim[1]-crop_dim[1])/2
        return reshaped_img[crop_amt[0]:crop_amt[0]+crop_dim[0], 
                            crop_amt[1]:crop_amt[1]+crop_dim[1]]

    def extract(self, image, features=['fc7'], new_dim=None, crop_dim=None):
        if new_dim is None:
            new_dim = self.new_shape
        if crop_dim is None:
            crop_dim = self.crop
        self.net.blobs['data'].data[...] = self.transformer.preprocess('data', 
                                            self.reshape_image(image, new_dim, crop_dim))
        _ = self.net.forward(end=features[-1])
        return [self.net.blobs[ft].data[0].tolist() for ft in features]

class PretrainedSVC:
    def __init__(self, model):
        self.svc = pickle.load(open(model, 'rb'))

    def predict(self, image_features):
        return self.svc.predict(image_features)

class MonaLisa:
    def __init__(self, models={}):
        self.models = models

    def add_model(self, name, feature_engine, pred_model):
        if name in self.models:
            print 'WARNING: model ' + name + ' already exists in MonaLisa'
        self.models[name] = (feature_engine, pred_model)

    def predict(self, image):
        ans = []
        for k, v in self.models.items():
            ft = v[0].extract(image)[0]
            ans.append(v[1].predict(ft)[0])
        return ans


def main():
    pass

"""
import caffe    
from mona_lisa import VGGFeatureExtractor, PretrainedSVC, MonaLisa
image_root = '/Users/ecsark/Documents/visualdb/project/wikiart/images/'
proj_root = '/Users/ecsark/Documents/visualdb/project/artwork-explorer/'
caffe_root = '/Users/ecsark/Documents/visualdb/caffe/'
model_weights = caffe_root + 'models/vgg/model.caffemodel'
model_def = caffe_root + 'models/vgg/deploy.prototxt'
vgg_ft = VGGFeatureExtractor(model_weights, model_def)
vgg_svc = PretrainedSVC(proj_root + 'model/svc_vgg_fc7.pk')
ml = MonaLisa()
ml.add_model('vgg_fc7', vgg_ft, vgg_svc)
image = caffe.io.load_image(image_root + 'salvador-dali_still-life-pulpo-y-scorpa.jpg')
print ml.predict(image)


"""
