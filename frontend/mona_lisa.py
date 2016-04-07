import numpy as np
import caffe
import cv2
from scipy.spatial import distance
from sklearn import svm
import cPickle as pickle
import os, sys
import subprocess
import math
import lmdb

class VGGFeatureExtractorCPP:
    def __init__(self, input_file, dst_image, output_dir, 
                model_def, model_weights, extract_feature_bin, feature_def):
        self.run_command = [extract_feature_bin, model_weights, model_def, feature_def, output_dir, '1', 'lmdb']
        self.output_dir = output_dir
        self.input_file = input_file
        self.dst_image = dst_image

    @staticmethod
    def read_from_db(db_dir):
        lmdb_env = lmdb.open(db_dir)
        lmdb_txn = lmdb_env.begin()
        lmdb_cursor = lmdb_txn.cursor()
        datum = caffe.proto.caffe_pb2.Datum()
        D = []
        for idx, (key, value) in enumerate(lmdb_cursor):
            datum.ParseFromString(value)
            data = caffe.io.datum_to_array(datum)
            D.append(data.flatten())

        lmdb_env.close()
        return D

    def extract(self, image):
        with open(self.dst_image, 'wb') as f:
            f.write(image)
        subprocess.call(['rm', '-rf', self.output_dir])
        ret = subprocess.call(self.run_command)
        if ret == 0:
            return self.read_from_db(self.output_dir)[0]
        else:
            print "Feature extraction failed"

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
        _ = self.net.forward()
        return [self.net.blobs[ft].data[0].tolist() for ft in features]

class PretrainedSVC:
    def __init__(self, model):
        self.svc = pickle.load(open(model, 'rb'))

    def predict(self, image_features):
        return self.svc.predict(image_features)

    def get_decision(self, fts):
        decisions = self.svc.decision_function(fts)
        n_classes = int((1+math.sqrt(1+8*len(decisions[0]))) / 2)
        scores = []
        idx = 0
        for i in range(n_classes):
            i_score = []
            for j in range(i):
                i_score.append(-scores[j][i])
            i_score.append(0)
            for k in range(n_classes - i - 1):
                i_score.append(decisions[0][idx])
                idx += 1
            scores.append(i_score)
        return scores

class ZeroScoreRecommender:
    def __init__(self, all_scores):
        all_scores = pickle.load(open(all_scores, 'rb'))
        self.score_rank_zero = [(np.array(decision)>0).sum(axis=1).tolist() for decision in all_scores]

    @staticmethod
    def findMostSimilar(X, x_query, k=6):
        dist = distance.cdist(X, np.array([x_query]), 'euclidean')
        return dist.flatten().argsort()[:k]

    def recommend(self, decision, k=6):
        k_sim_idx = self.findMostSimilar(self.score_rank_zero, (np.array(decision)>0).sum(axis=1).tolist(), k)
        return k_sim_idx


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
