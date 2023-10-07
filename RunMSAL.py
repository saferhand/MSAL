from river import stream
import os
from MSAL import Evaluator
from. import MSAL


def run_experiments_one_pm(data_list, compared_method_list, parallel_experiment_times, output_folder,parval):

    tempname = 'r' + str(parval[0]) + \
               'u' + str(parval[1]) + \
               'c' + str(parval[2]) + \
               'i' + str(parval[3]) + \
               'd' + str(parval[4])

    for arff_data in data_list:
        for algorithm in compared_method_list:
            for times in range(3, parallel_experiment_times):
                result_csv_name = arff_data + '_'+ algorithm + tempname + '_times_' + str(times) + '_0.csv'
                print(result_csv_name)

                if result_csv_name in os.listdir(output_folder):
                    continue

                path = data_folder + arff_data
                X_y = stream.iter_arff(path, target='class')

                expmodel = MSAL(X_y,
                                parval[0],
                                parval[1],
                                parval[2],
                                parval[3],
                                parval[4]
                                )

                labellist, predlist, allist, ddlist, nspredlist = expmodel.learning_procedure()
                eva = Evaluator()
                resulfdf = eva.evaluate_accuracy_score(labellist, predlist, allist, nspredlist, 100)
                resulfdf.to_csv(output_folder + result_csv_name, index= False)



parameterslist = []
parameterslist.append([50, 0.10, 0.15, 50, 0.1])
print(len((parameterslist)))

data_folder = 'your_data_path'
data_list = os.listdir(data_folder)

data_list = [
    'data_name.arff',
]

parallel_experiment_times = 5
output_folder = 'your_output_file_path'
compared_method_list =[
    'MSAL'
]
for pa in parameterslist:
    run_experiments_one_pm(data_list, compared_method_list, parallel_experiment_times, output_folder, pa)