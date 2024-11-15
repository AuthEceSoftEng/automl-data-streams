import copy

import pandas as pd


def result_extractor (experiment_no: int,pipeline_no: int,y_real_temp, y_predicted_temp, data_drifts_temp, concept_drifts_temp):
    y_real_temp_copy = copy.deepcopy(y_real_temp)
    y_predicted_temp_copy = copy.deepcopy(y_predicted_temp)
    concept_drifts_temp_copy = copy.deepcopy(concept_drifts_temp)
    data_drifts_temp_copy = copy.deepcopy(data_drifts_temp)
    concept_drifts_temp_copy.extend([None] * (len(y_real_temp) - len(concept_drifts_temp_copy)))
    data_drifts_temp_copy.extend([None] * (len(y_real_temp) - len(data_drifts_temp_copy)))

    frame = pd.DataFrame({
            'y_real': y_real_temp_copy,
            'y_pred': y_predicted_temp_copy,
            'data_drift': data_drifts_temp_copy,
            'concept_drift': concept_drifts_temp_copy
        })

    frame.to_csv(f'Results/experiment{experiment_no}_pipeline{pipeline_no}_results.csv', index=False)