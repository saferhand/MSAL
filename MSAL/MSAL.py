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
from adapt.instance_based import KLIEP
import numpy as np

import sys
import io

class MSAL:

    def __init__(self, datastream, al_ini_num, al_rd_thd, al_budgets,
                 window_size, weight_adjust_step):

        self.TARGET_DOMAIN_INDEX = 0

        self.X_y = datastream
        self.al_ini_num = al_ini_num
        self.AL_RANDOM_THRESHOLD = al_rd_thd
        self.AL_Budgets = al_budgets
        self.window_size = window_size
        self.weight_adjust_step = weight_adjust_step

        '''
        :param datastream: 待实验的multi-source数据流
        :param al_rd_thd: 主动学习随机策略的阈值
        :param AL_Budgets: 主动学习标签率上限
        :param al_ini_num: 初始阶段主动学习的样本数量
        '''


    def silence_print_output(func):
        """
        Decorator to silence print outputs of a function.

        Parameters:
        - func: Function whose print output needs to be silenced.

        Returns:
        - Wrapped function.
        """

        def wrapper(*args, **kwargs):
             
            original_stdout = sys.stdout

             
            sys.stdout = io.StringIO()

             
            result = func(*args, **kwargs)

             
            sys.stdout = original_stdout

            return result

        return wrapper

    def dict_list_to_matrix(self,window_data):
         
        matrix = []
         
        for item in window_data:
            row = [item[key] for key in sorted(item.keys())]
            matrix.append(row)

         
        return np.array(matrix)

     
    @silence_print_output
    def perform_KLIEP(self, Bs, Bt):

        BS_array = self.dict_list_to_matrix(Bs)
        Bt_array = self.dict_list_to_matrix(Bt)

         
         
        kliep = KLIEP(kernel="rbf", gamma=[10 ** (i - 4) for i in range(6)], random_state=0)
        kliep_weights = kliep.fit_weights(BS_array, Bt_array)
        return kliep_weights

    def creat_model_for_domain(self):
        model = ensemble.BaggingClassifier(
            model=(
                preprocessing.StandardScaler() |
                tree.HoeffdingTreeClassifier()
            ),
            n_models=10,
            seed=2
        )
        return model

    def creat_ddmodel_for_domain(self):
        ddmodel = HDDM_A()
        return ddmodel

    def get_votes_for_instance(self, x, model):
        y_pred = collections.Counter()
        for bsm in model:
            y_pred.update(bsm.predict_proba_one(x))
        total = sum(y_pred.values())
        if total > 0:
            result = {label: proba / total for label, proba in y_pred.items()}
            return result
        return y_pred

    def updata_Weight(self, ms_ensemble, ms_weight, x, y):
        for keys, values in ms_ensemble.items():
            for j, basemodel in enumerate(values):   
                for k, baseclf in enumerate(basemodel):
                    votes = baseclf.predict_proba_one(x)
                    if len(votes) != 2:
                        votes = {0: 0.0, 1: 0.0}
                    if max(votes, key=votes.get) == y:
                        ms_weight[keys][j][k] = ms_weight[keys][j][k] * (1 + self.weight_adjust_step)
                    else:
                        ms_weight[keys][j][k] = ms_weight[keys][j][k] * (1 - self.weight_adjust_step)
        return ms_weight

    def updata_Weights_fake(self, ms_fake_ensemble, ms_fake_ensemble_weights, x, y):
        for keys, values in ms_fake_ensemble.items():
            for j, basemodel in enumerate(values):   
                for k, baseclf in enumerate(basemodel):
                    votes = baseclf.predict_proba_one(x)
                    if len(votes) != 2:
                        votes = {0: 0.0, 1: 0.0}
                    if max(votes, key=votes.get) == y:
                        ms_fake_ensemble_weights[keys][j][k] = ms_fake_ensemble_weights[keys][j][k] * (1 + self.weight_adjust_step)
                    else:
                        ms_fake_ensemble_weights[keys][j][k] = ms_fake_ensemble_weights[keys][j][k] * (1 - self.weight_adjust_step)

        return ms_fake_ensemble_weights

    def select_samples_by_weight(self, weights, samples, labels, k):
        """
        Select k samples based on the smallest weights and where the corresponding label is 0.

        Parameters:
        - weights: List of weights (floats).
        - samples: List of samples (dicts).
        - labels: List of labaled (0 or 1).
        - k: Number of samples to select.

        Returns:
        - List of indices of the selected samples.
        """

         
        if not (len(weights) == len(samples) == len(labels)):
            raise ValueError("All lists must have the same length.")

         
        sorted_indices = np.argsort(weights)

         
        chosen_indices = []

         
        count = 0

         
        for index in sorted_indices:
             
            if labels[index] == 0:
                 
                chosen_indices.append(index)
                 
                count += 1
             
            if count == k:
                break

        return chosen_indices

    def learning_procedure(self):

         
        target_domain_label = []
        target_domain_pred = []
        target_domain_joint_pred = []
        al_labeled_window = []
        ddm_list = []

        processed_sample = 0
        target_domain_sample_num = 0

         
        ms_ensemble = {}
        ms_ddmodel = {}
        ms_weight = {}

         
        ms_fake_ensemble = {}
        ms_fake_ensemble_weights = {}

        source_window = []
        source_window_label = []     

        target_ref_window = []   
        target_test_window = []  

        target_test_window_labeled = []      
        target_test_window_labels = []       

         
        target_window_count = 0

        for x, y in self.X_y:

            processed_sample = processed_sample + 1
             
            domain_id_value = 0
            domain_id_keys = ''
            for keys,values in x.items():
                domain_id_value = int(values)
                domain_id_keys = keys
                break
            x.pop(domain_id_keys)

             
            if domain_id_value not in ms_ensemble.keys():
                 
                ms_ensemble[domain_id_value] = []
                ms_ensemble[domain_id_value].append(self.creat_model_for_domain())
                 
                ms_ddmodel[domain_id_value] = []
                ms_ddmodel[domain_id_value].append(self.creat_ddmodel_for_domain())
                 
                num_base_classifier = ms_ensemble[domain_id_value][-1].n_models
                ms_weight[domain_id_value] = []
                ms_weight[domain_id_value].append([i-i+1.0 for i in range(0, num_base_classifier)])

                 
                if domain_id_value != self.TARGET_DOMAIN_INDEX:
                     
                    ms_fake_ensemble[domain_id_value] = []
                    ms_fake_ensemble[domain_id_value].append(self.creat_model_for_domain())
                    ms_fake_ensemble_weights[domain_id_value] = []
                    ms_fake_ensemble_weights[domain_id_value].append([i - i + 1.0 for i in range(0, num_base_classifier)])

             
            if domain_id_value == self.TARGET_DOMAIN_INDEX:
                al_labeled_flag = False
                sigma_value = random.random()
                if target_domain_sample_num <= self.al_ini_num:  
                    al_labeled_flag = True
                    target_test_window_labeled.append(1)         
                elif sigma_value < (self.AL_RANDOM_THRESHOLD):
                    al_labeled_flag = True
                    target_test_window_labeled.append(1)         
                else:
                    target_test_window_labeled.append(0)         

             
            if domain_id_value == self.TARGET_DOMAIN_INDEX:

                 
                target_window_count = target_window_count + 1
                 
                if target_window_count>=self.window_size:        
                    target_window_count = 0

                 
                if len(target_ref_window)<self.window_size:
                    target_ref_window.append(x)
                target_test_window.append(x)
                target_test_window_labels.append(y)

                 
                if len(target_test_window) > self.window_size:   
                    del target_test_window[0]
                    del target_test_window_labels[0]

            else:
                 
                source_window.append(x)
                source_window_label.append(y)
                if len(source_window) > self.window_size:
                    del source_window[0]
                    del source_window_label[0]

             
            if domain_id_value == self.TARGET_DOMAIN_INDEX:  
                temp_pred = self.get_votes_for_instance(x, ms_ensemble[domain_id_value][-1])
                 
                if al_labeled_flag == True:
                    if len(temp_pred) == 0:
                        correctedpred_r = 1
                    if temp_pred == y:
                        correctedpred_r = 0      
                    else:
                        correctedpred_r = 1      
                    in_drift, in_warning = ms_ddmodel[domain_id_value][-1].update(correctedpred_r)
            else:  
                temp_pred = self.get_votes_for_instance(x, ms_ensemble[domain_id_value][-1])
                if len(temp_pred)==0:
                    correctedpred = 1
                if temp_pred == y:
                    correctedpred = 0
                else:
                    correctedpred = 1
                in_drift, in_warning = ms_ddmodel[domain_id_value][-1].update(correctedpred)

             
            if in_drift and domain_id_value == self.TARGET_DOMAIN_INDEX:         
                 
                print(f"MS Change detected at index {processed_sample}, in domain: {domain_id_value}")
                ms_ensemble[domain_id_value].append(self.creat_model_for_domain())
                num_base_classifier = ms_ensemble[domain_id_value][-1].n_models
                ms_weight[domain_id_value].append([i-i+1 for i in range(0, num_base_classifier)])
                for keys, values in ms_weight.items():   
                    for j, baseem in enumerate(values):   
                        for k, baseclf in enumerate(baseem):   
                            ms_weight[keys][j][k] = 1.0
                target_ref_window = target_test_window
                target_test_window = []
                target_test_window.append(x)
                target_window_count = 1

             
            elif in_drift and domain_id_value != self.TARGET_DOMAIN_INDEX:       
                 
                ms_ensemble[domain_id_value].append(self.creat_model_for_domain())
                num_base_classifier = ms_ensemble[domain_id_value][-1].n_models
                ms_weight[domain_id_value].append([i - i + 1 for i in range(0, num_base_classifier)])
                 
                for keys, values in ms_weight.items():   
                    for j, baseem in enumerate(values):   
                        for k, baseclf in enumerate(baseem):   
                            if keys == domain_id_value:
                                ms_weight[keys][j][k] = 1.0

             
            if target_window_count == 0 and domain_id_value == self.TARGET_DOMAIN_INDEX:

                 
                source_target_weights = self.perform_KLIEP(source_window, target_test_window)
                source_target_weights /= source_target_weights.sum()
                n = self.window_size
                chosen_samples_window_indices = np.random.choice(len(source_window), size=n, p=source_target_weights)
                chosen_samples = [source_window[i] for i in chosen_samples_window_indices]
                chosen_labels = [source_window_label[i] for i in chosen_samples_window_indices]

                 
                for source_id in ms_fake_ensemble.keys():
                    for samplex, sampley in zip(chosen_samples, chosen_labels):
                        ms_fake_ensemble[source_id][-1].learn_one(samplex, sampley)

                 
                target_reftest_weights = self.perform_KLIEP(target_test_window, target_ref_window)

                 
                 
                sample_indices = self.select_samples_by_weight(target_reftest_weights,
                                                               target_test_window,
                                                               target_test_window_labeled,
                                                               math.floor((self.AL_Budgets-self.AL_RANDOM_THRESHOLD)*self.window_size))
                target_test_window_labeled = []
                for indics_val in  sample_indices:
                    ms_ensemble[self.TARGET_DOMAIN_INDEX][-1].learn_one(
                        target_test_window[indics_val], target_test_window_labels[indics_val])
                    self.updata_Weight(ms_ensemble, ms_weight,
                                       target_test_window[indics_val], target_test_window_labels[indics_val])
                    self.updata_Weights_fake(ms_fake_ensemble, ms_fake_ensemble_weights,
                                             target_test_window[indics_val], target_test_window_labels[indics_val])

             
            if domain_id_value!= self.TARGET_DOMAIN_INDEX:
                ms_ensemble[domain_id_value][-1].learn_one(x, y)
            else:
                 
                if in_drift:
                    ddm_list.append(2)
                elif in_warning:
                    ddm_list.append(1)
                else:
                    ddm_list.append(0)

                target_domain_sample_num = target_domain_sample_num + 1
                target_domain_label.append(y)

                 
                predddy = self.get_votes_for_instance(x, ms_ensemble[domain_id_value][-1])
                if len(predddy) == 0:
                    predddy = y
                else:
                    predddy = max(predddy, key=predddy.get)
                target_domain_pred.append(predddy)

                 
                y_joint_pred = collections.Counter()
                for keys,values in ms_ensemble.items():
                    for j, basemodel in enumerate(values):  
                        for k,baseclf in enumerate(basemodel):
                            temp_weight = ms_weight[keys][j][k]
                            votes = baseclf.predict_proba_one(x)
                            for key in votes:
                                votes[key] *= temp_weight
                            y_joint_pred.update(votes)

                 
                for keys,values in ms_fake_ensemble.items():
                    for j, basemodel in enumerate(values):  
                        for k,baseclf in enumerate(basemodel):
                            temp_weight = ms_fake_ensemble_weights[keys][j][k]
                            votes = baseclf.predict_proba_one(x)
                            for key in votes:
                                votes[key] *= temp_weight
                            y_joint_pred.update(votes)
                if len(y_joint_pred) == 0:
                    y_joint_pred = y
                else:
                    y_joint_pred = max(y_joint_pred, key=y_joint_pred.get)
                target_domain_joint_pred.append(y_joint_pred)

                if al_labeled_flag:
                    al_labeled_window.append(1)
                    ms_ensemble[domain_id_value][-1].learn_one(x, y)
                    ms_weight = self.updata_Weight(ms_ensemble, ms_weight, x, y)
                    ms_fake_ensemble_weights = self.updata_Weights_fake(ms_fake_ensemble, ms_fake_ensemble_weights, x, y)
                else:
                    al_labeled_window.append(0)

        print("Target domain accuracy",accuracy_score(target_domain_label, target_domain_pred))
        print("Target domain joint pred accuracy",accuracy_score(target_domain_label, target_domain_joint_pred))
        print("Target domain ensemble number",len(ms_ensemble[self.TARGET_DOMAIN_INDEX]))
        print("Acitve learning budgets:", sum(al_labeled_window) / len(al_labeled_window))

        return target_domain_label, target_domain_joint_pred, al_labeled_window, ddm_list, target_domain_pred