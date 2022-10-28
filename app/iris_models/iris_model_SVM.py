import pickle
import time
import pandas as pd
from iris_models.iris_scaler import scale_inputs

def predict_SVM(input_features, input_features_dict, iris_types):
    
    start_time = time.time()

    score_SVM = pd.read_csv('iris_data//iris_report_SVM.csv').set_index('class')
    
    accuracy = score_SVM.loc['accuracy'].mean()
    support = score_SVM.iloc[-1,-1]
    
    print('\nAccuracy of the SVM model:\n', accuracy)
    print('\nTest Support of the SVM model:\n', support)
    print('\nScore of the SVM model:\n', score_SVM.iloc[0:3,:])    
    
    SVM_model = pickle.load(open("iris_models//iris_model_SVM.pkl", "rb"))
    
    scale_inputs_returned_values = scale_inputs(input_features)
    
    input_features_scaled = scale_inputs_returned_values['input_features_scaled']
    input_features_scaled_dict = scale_inputs_returned_values['input_features_scaled_dict']
        
    iris_pred = int(SVM_model.predict(input_features_scaled))
    
    prediction = (iris_pred, iris_types[iris_pred])
    print('\nPrediction with SVM model:\n',prediction)

    probabil = SVM_model.predict_proba(input_features_scaled) 
    probab_setosa = round(float(probabil[:,0]),6)
    probab_versicolor = round(float(probabil[:,1]),6)
    probab_virginica = round(float(probabil[:,2]),6)
          
    print('\nProbabilities with SVM model:',
          '\nprob. setosa:',probab_setosa,
          '\nprob. versicolor:',probab_versicolor,
          '\nprob. virginica:',probab_virginica)
    
    exec_time = time.time() - start_time
    print('\nExecution time with SVM model:\n',round(exec_time,6),'\n')
    
    return {'model_accuracy': accuracy,
            'test_support': support,
            'iris types': iris_types,
            'scores_setosa': score_SVM.loc['0'],
            'scores_versicolor': score_SVM.loc['1'],
            'scores_virginica': score_SVM.loc['2'],
            'input_features:': input_features_dict,
            'input_features_scaled': input_features_scaled_dict,
            'prediction': prediction, 
            'probab_setosa': probab_setosa, 
            'probab_versicolor': probab_versicolor,
            'probab_virginica': probab_virginica,
            'exec_time': exec_time}