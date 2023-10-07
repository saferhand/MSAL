import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score


class Evaluator:
    """Electricity prices in New South Wales.

    This is a binary classification task, where the goal is to predict if the price of electricity
    will go up or down.

    This data was collected from the Australian New South Wales Electricity Market. In this market,
    prices are not fixed and are affected by demand and supply of the market. They are set every
    five minutes. Electricity transfers to/from the neighboring state of Victoria were done to
    alleviate fluctuations.

    References
    ----------
    [^1]: [SPLICE-2 Comparative Evaluation: Electricity Pricing](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.12.9405)
    [^2]: [DataHub description](https://datahub.io/machine-learning/electricity#readme)

    """


    @classmethod
    def evaluate_accuracy_score(self, real_label, pred_label, al_label_list, ns_pred_label, sample_frequency):

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
        anaresultlist_ns = []
        for i in range(1, round(length / ana_peroid) + 1):

            analength = i * ana_peroid
            anaresult = accuracy_score(real_label[0:analength], pred_label[0:analength])
            anaallabel = sum(al_label_list[0:analength]) / len(al_label_list[0:analength])
            anaresult_ns = accuracy_score(real_label[0:analength], ns_pred_label[0:analength])

            analengthlist.append(analength)
            anaresultlist.append(anaresult*100)
            anaallabellist.append(anaallabel)
            anaresultlist_ns.append(anaresult_ns)

        if analength < length:

            analength = length
            anaresult = accuracy_score(real_label[0:analength], pred_label[0:analength])
            anaresult_ns = accuracy_score(real_label[0:analength], ns_pred_label[0:analength])

            analengthlist.append(analength)
            anaresultlist.append(anaresult*100)
            anaallabellist.append(anaallabel)
            anaresultlist_ns.append(anaresult_ns)

        result_csv['learning evaluation instances'] = analengthlist
        result_csv['classifications correct (percent)'] = anaresultlist
        result_csv['active learning budguts'] = anaallabellist
        result_csv['classifications correct without source (percent)'] = anaresultlist_ns

        return result_csv