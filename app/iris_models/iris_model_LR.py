import pickle
import time
import pandas as pd
from iris_models.iris_scaler import scale_inputs

def predict_LR(input_features, input_features_dict, iris_types):
    
    start_time = time.time()

    score_LR = pd.read_csv('iris_data//iris_report_LR.csv').set_index('class')
    
    accuracy = score_LR.loc['accuracy'].mean()
    support = score_LR.iloc[-1,-1]
    
    print('\nAccuracy of the LR model:\n', accuracy)
    print('\nTest Support of the LR model:\n', support)
    print('\nScore of the LR model:\n', score_LR.iloc[0:3,:])    
    
    LR_model = pickle.load(open("iris_models//iris_model_LR.pkl", "rb"))
    
    scale_inputs_returned_values = scale_inputs(input_features)
    
    input_features_scaled = scale_inputs_returned_values['input_features_scaled']
    input_features_scaled_dict = scale_inputs_returned_values['input_features_scaled_dict']
        
    iris_pred = int(LR_model.predict(input_features_scaled))
    
    prediction = (iris_pred, iris_types[iris_pred])
    print('\nPrediction with LR model:\n',prediction)

    probabil = LR_model.predict_proba(input_features_scaled) 
    probab_setosa = round(float(probabil[:,0]),6)
    probab_versicolor = round(float(probabil[:,1]),6)
    probab_virginica = round(float(probabil[:,2]),6)
          
    print('\nProbabilities with LR model:',
          '\nprob. setosa:',probab_setosa,
          '\nprob. versicolor:',probab_versicolor,
          '\nprob. virginica:',probab_virginica)
    
    exec_time = time.time() - start_time
    print('\nExecution time with LR model:\n',round(exec_time,6),'\n')
    
    return {'model_accuracy': accuracy,
            'test_support': support,
            'iris types': iris_types,
            'scores_setosa': score_LR.loc['0'],
            'scores_versicolor': score_LR.loc['1'],
            'scores_virginica': score_LR.loc['2'],
            'input_features:': input_features_dict,
            'input_features_scaled': input_features_scaled_dict,
            'prediction': prediction, 
            'probab_setosa': probab_setosa, 
            'probab_versicolor': probab_versicolor,
            'probab_virginica': probab_virginica,
            'exec_time': exec_time}