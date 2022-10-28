import pickle
import time
import pandas as pd
import numpy as np
from tensorflow import keras
from iris_models.iris_scaler import scale_inputs

def predict_NN(input_features, input_features_dict, iris_types):
    
    start_time = time.time()

    score_NN = pd.read_csv('iris_data//iris_report_NN.csv').set_index('class')
    
    accuracy = score_NN.loc['accuracy'].mean()
    support = score_NN.iloc[-1,-1]
    
    print('\nAccuracy of the NN model:\n', accuracy)
    print('\nTest Support of the NN model:\n', support)
    print('\nScore of the NN model:\n', score_NN.iloc[0:3,:])    
    
    NN_model = keras.models.load_model("iris_models//iris_model_NN.h5")
    
    scale_inputs_returned_values = scale_inputs(input_features)
    
    input_features_scaled = scale_inputs_returned_values['input_features_scaled']
    input_features_scaled_dict = scale_inputs_returned_values['input_features_scaled_dict']
        
    #retrieving prediction from probabilities
    probabil = NN_model.predict(input_features_scaled)
    iris_pred = int(np.argmax(probabil, axis=1))
#     print('\niris_pred:\n',iris_pred)  ##########
    
    prediction = (iris_pred, iris_types[iris_pred])
    print('\nPrediction with NN model:\n',prediction)

#     probabil = NN_model.predict_proba(input_features_scaled) 
    probab_setosa = round(float(probabil[:,0]),6)
    probab_versicolor = round(float(probabil[:,1]),6)
    probab_virginica = round(float(probabil[:,2]),6)
          
    print('\nProbabilities with NN model:',
          '\nprob. setosa:',probab_setosa,
          '\nprob. versicolor:',probab_versicolor,
          '\nprob. virginica:',probab_virginica)
    
    exec_time = time.time() - start_time
    print('\nExecution time with NN model:\n',round(exec_time,6),'\n')
    
    return {'model_accuracy': accuracy,
            'test_support': support,
            'iris types': iris_types,
            'scores_setosa': score_NN.loc['0'],
            'scores_versicolor': score_NN.loc['1'],
            'scores_virginica': score_NN.loc['2'],
            'input_features:': input_features_dict,
            'input_features_scaled': input_features_scaled_dict,
            'prediction': prediction, 
            'probab_setosa': probab_setosa, 
            'probab_versicolor': probab_versicolor,
            'probab_virginica': probab_virginica,
            'exec_time': exec_time}