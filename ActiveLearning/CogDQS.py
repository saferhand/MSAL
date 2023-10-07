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
from scipy.spatial import distance
import numpy as np

class CogDQS:

    def __init__(self, datastream,
                 al_rd_thd,
                 al_ini_num,
                 al_un_thd,
                 al_un_step,
                 cognition_window_length,
                 local_density_threshold,
                 al_budgets,
                 budget_control_flag,
                 al_strategy,
                 budget_evulate_window_size):

        self.X_y = datastream                       
        self.AL_RANDOM_THRESHOLD = al_rd_thd        
        self.al_ini_num = al_ini_num                
        self.al_un_thd = al_un_thd                  
        self.al_un_step = al_un_step                
        self.cognition_window_length = cognition_window_length  
        self.local_density_threshold = local_density_threshold  
        self.al_budgets = al_budgets                            
        self.budget_control_flag = budget_control_flag          
        self.al_strategy = al_strategy                          
        self.budget_evulate_window_size = budget_evulate_window_size 

    def creat_model_for_domain(self):
        model = ensemble.BaggingClassifier(
            model=(
                preprocessing.StandardScaler() |
                tree.HoeffdingTreeClassifier()
            ),
            n_models=10,
            seed=random.randint(0,99)
        )
        return model

    def creat_ddmodel_for_domain(self):
        ddmodel = HDDM_A()
        return ddmodel

    def calculate_local_density(self, xi, cw):

        min_dis = 999999
        for xj in cw:

            

            

                
                
                
                


            dis_euclidean = distance.euclidean(xi['data'], xj['data'])
            if dis_euclidean < min_dis:
                min_dis = dis_euclidean
        

        

        
        return min_dis


    def ini_create_dict_for_sample(self, x, t):

        sample_dict = {}
        sample_dict['time'] = t
        sample_dict['data'] = x
        sample_dict['s_memory_strength'] = 1
        sample_dict['lambda'] = 0
        sample_dict['tau'] = t
        sample_dict['f_fading_factor'] = 1
        sample_dict['minDIS'] = 0
        
        
        
        
        
        
        
        
        
        return sample_dict



    def learning_procedure(self):

        real_label = []
        pred_label = []
        al_labeled_window = []
        ddm_list = []

        cognition_window = []

        processed_sample = 0

        drift_detector = self.creat_ddmodel_for_domain()
        model = self.creat_model_for_domain()

        B_budget_evu = 0

        for x, y in self.X_y:

            
            processed_sample = processed_sample + 1

            x_data = np.array(list(x.values())).astype(float)
            x_dict = self.ini_create_dict_for_sample(x_data, processed_sample)
            real_label.append(y)

            ld_xi = 0
            if len(cognition_window) > 0:
                
                ld_xi = 0
                for xj in cognition_window:
                    dis_euclidean = distance.euclidean(x_dict['data'], xj['data'])
                    if xj['minDIS'] == 0:
                        xj['minDIS'] = dis_euclidean
                    elif xj['minDIS'] > dis_euclidean:
                        xj['minDIS'] = dis_euclidean
                        xj['tau'] = x_dict['time']
                        xj['lambda'] = xj['lambda'] + 1
                        ld_xi = ld_xi + 1
                    

                
                for xj in cognition_window:
                    xj['f_fading_factor'] = 1/(xj['f_fading_factor']+1)
                    xj['s_memory_strength'] = math.exp(-1*xj['f_fading_factor']*(x_dict['time'] - xj['tau']))

            if len(cognition_window) >= self.cognition_window_length:
                delete_id = -1
                min_s_value = 99999
                for id, xj in enumerate(cognition_window):
                    if xj['s_memory_strength'] < min_s_value:
                        min_s_value = xj['s_memory_strength']
                        delete_id = id
                cognition_window.pop(delete_id)
            
            if len(cognition_window) < self.cognition_window_length:
                cognition_window.append(x_dict)

            if len(al_labeled_window) == 0:
                current_labeling_cost = 0
            else:
                current_labeling_cost = sum(al_labeled_window) / len(al_labeled_window)

            al_strategy = self.al_strategy
            al_labeled_flag = False

            
            

            if self.budget_control_flag :  
                current_labeling_cost = 0   

            

            
            if al_strategy == 'ran' and ld_xi>0:
                sigma_value = random.random()
                if (sigma_value < self.AL_RANDOM_THRESHOLD or
                processed_sample < self.al_ini_num) \
                    and current_labeling_cost < self.AL_RANDOM_THRESHOLD:
                    al_labeled_flag = True
            
            elif al_strategy == 'fu' and ld_xi>0:
                y_pred = model.predict_proba_one(x)
                if len(y_pred) <= 1:
                    max_prob = 1.0
                else:
                    max_prob = max(y_pred.values())
                if (max_prob < self.al_un_thd or processed_sample < self.al_ini_num) and current_labeling_cost < self.al_budgets:
                    al_labeled_flag = True
            
            elif al_strategy == 'vu' and ld_xi>0:
                y_pred = model.predict_proba_one(x)
                if len(y_pred) <= 1:
                    max_prob = 1.0
                else:
                    max_prob = max(y_pred.values())
                if (max_prob < self.al_un_thd or processed_sample < self.al_ini_num) and current_labeling_cost < self.al_budgets:
                    al_labeled_flag = True
                    self.al_un_thd = self.al_un_thd * (1 - self.al_un_step)
                else:
                    self.al_un_thd = self.al_un_thd * (1 + self.al_un_step)
            
            elif al_strategy == 'rvu' and ld_xi>0:
                y_pred = model.predict_proba_one(x)
                if len(y_pred) <=1:
                    max_prob = 1.0
                else:
                    max_prob = max(y_pred.values())
                max_prob = max_prob/(np.random.normal(loc=0.0, scale=1.0, size=None) + 1)
                if (max_prob < self.al_un_thd or processed_sample < self.al_ini_num) and current_labeling_cost < self.al_budgets:
                    al_labeled_flag = True
                    self.al_un_thd = self.al_un_thd * (1 - self.al_un_step)
                else:
                    self.al_un_thd = self.al_un_thd * (1 + self.al_un_step)

            if model.predict_one(x) == y:
                if al_labeled_flag:
                    drift_detector.update(1)
                pred_label.append(y)
            else:
                if al_labeled_flag:
                    drift_detector.update(0)
                if model.predict_one(x)!= None:
                    pred_label.append(model.predict_one(x))
                else:
                    pred_label.append(-1)

            if processed_sample< self.al_ini_num:
                al_labeled_flag = True

            if al_labeled_flag:
                model.learn_one(x,y)
                al_labeled_window.append(1)
            else:
                al_labeled_window.append(0)

            if len(al_labeled_window) == 0:
                current_labeling_cost = 0
            else:
                current_labeling_cost = sum(al_labeled_window) / len(al_labeled_window)

            
            
            
            if len(al_labeled_window) > self.budget_evulate_window_size:
                temp_al_labeled_window = al_labeled_window[len(al_labeled_window)-self.budget_evulate_window_size:]
                v_budget_evu = sum(temp_al_labeled_window)
            else:
                v_budget_evu = sum(al_labeled_window)
            if al_labeled_flag:
                temp_value = 1
            else:
                temp_value = 0
            v_budget_evu = v_budget_evu*(self.budget_evulate_window_size-1)/self.budget_evulate_window_size + temp_value
            B_budget_evu = v_budget_evu/self.budget_evulate_window_size
            

        if drift_detector.change_detected:
            
            print(f'Change detected at index {processed_sample}')
            drift_detector.reset()
            model = self.creat_model_for_domain()

        print("Accuracy",accuracy_score(real_label, pred_label))
        print("Acitve learning budgets:", sum(al_labeled_window) / len(al_labeled_window))

        return real_label, \
               pred_label, \
               al_labeled_window, \
               ddm_list
