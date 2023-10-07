from river import stream
from river import ensemble
from river import preprocessing
from river import tree
from sklearn.metrics import accuracy_score
from river.drift import HDDM_A
from river.drift import DDM
import collections
import random
import math
from scipy.stats import ks_2samp

class OB:

    def __init__(self, datastream, runtimes):
        self.X_y = datastream
        self.seed = runtimes

    def creat_model_for_domain(self):
        model = ensemble.BaggingClassifier(
            model=(
                preprocessing.StandardScaler() |
                tree.HoeffdingTreeClassifier()
            ),
            n_models=10,
            seed = self.seed
        )
        return model

    def creat_ddmodel_for_domain(self):
        ddmodel = HDDM_A()
        return ddmodel

    def learning_procedure(self):

        real_label = []
        pred_label = []
        al_labeled_window = []
        ddm_list = []

        processed_sample = 0

        
        model = self.creat_model_for_domain()

        for x, y in self.X_y:
            processed_sample = processed_sample + 1
            
            

            real_label.append(y)

            if model.predict_one(x) == y:
                
                
                pred_label.append(y)
            else:
                
                
                if model.predict_one(x)!= None:
                    pred_label.append(model.predict_one(x))
                else:
                    pred_label.append(-1)

            
            model.learn_one(x,y)
            al_labeled_window.append(1)
        
            

        
        
        
        
        

        print("Accuracy",accuracy_score(real_label, pred_label))
        

        return real_label, \
               pred_label, \
               al_labeled_window, \
               ddm_list
