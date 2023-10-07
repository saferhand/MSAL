import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score


class EvaluatorAL:

    @classmethod
    def evaluate_accuracy_score(self, real_label, pred_label, al_label_list, sample_frequency):

        """ evaluate_accuracy_score the results
            输入:
            1. real label list
            2. pred label list
            3. active learning labeled list

            输出:
            Dataframe len(real label list) * 3
            col0: 'learning evaluation instances'
            col1: 'classifications correct (percent)'
            col2: 'active learning budguts'

        """

        length = len(real_label)
        ana_peroid = sample_frequency
        result_csv = []
        result_csv = pd.DataFrame(result_csv)

        analengthlist = []
        anaresultlist = []
        anaallabellist = []

        for i in range(1, round(length / ana_peroid) + 1):

            analength = i * ana_peroid
            anaresult = accuracy_score(real_label[0:analength], pred_label[0:analength])
            anaallabel = sum(al_label_list[0:analength]) / len(al_label_list[0:analength])

            analengthlist.append(analength)
            anaresultlist.append(anaresult*100)
            anaallabellist.append(anaallabel)


        if analength < length:

            analength = length
            anaresult = accuracy_score(real_label[0:analength], pred_label[0:analength])
            analengthlist.append(analength)
            anaresultlist.append(anaresult*100)
            anaallabellist.append(anaallabel)

        result_csv['learning evaluation instances'] = analengthlist
        result_csv['classifications correct (percent)'] = anaresultlist
        result_csv['active learning budguts'] = anaallabellist

        return result_csv