import pickle
import time
import pandas as pd

def predict_KM(input_features, input_features_dict, iris_types):
    
    start_time = time.time()

    score_KM = pd.read_csv('iris_data//iris_report_KM.csv').set_index('class')
    
    accuracy = score_KM.loc['accuracy'].mean()
    support = score_KM.iloc[-1,-1]
    
    print('\nAccuracy of the KM model:\n', accuracy)
    print('\nTest Support of the KM model:\n', support)
    print('\nScore of the KM model:\n', score_KM.iloc[0:3,:])    
    
    KM_model = pickle.load(open("iris_models//iris_model_KM.pkl", "rb"))
        
    iris_pred = int(KM_model.predict(input_features))
    
    prediction = (iris_pred, iris_types[iris_pred])
    print('\nPrediction with KM model:\n',prediction)
    
          
    print('\nProbabilities with KM model:',
          '\nprob. setosa:','N/A',
          '\nprob. versicolor:','N/A',
          '\nprob. virginica:','N/A')
    
    exec_time = time.time() - start_time
    print('\nExecution time with KM model:\n',round(exec_time,6),'\n')
    
    return {'model_accuracy': accuracy,
            'test_support': support,
            'iris types': iris_types,
            'scores_setosa': score_KM.loc['0'],
            'scores_versicolor': score_KM.loc['1'],
            'scores_virginica': score_KM.loc['2'],
            'input_features:': input_features_dict,
            'prediction': prediction, 
            'probab_setosa': 'N/A', 
            'probab_versicolor': 'N/A',
            'probab_virginica': 'N/A',
            'exec_time': exec_time}