import pickle
import time
import pandas as pd

def predict_GB(input_features, input_features_dict, iris_types):
    
    start_time = time.time()

    score_GB = pd.read_csv('iris_data//iris_report_GB.csv').set_index('class')
    
    accuracy = score_GB.loc['accuracy'].mean()
    support = score_GB.iloc[-1,-1]
    
    print('\nAccuracy of the GB model:\n', accuracy)
    print('\nTest Support of the GB model:\n', support)
    print('\nScore of the GB model:\n', score_GB.iloc[0:3,:])    
    
    GB_model = pickle.load(open("iris_models//iris_model_GB.pkl", "rb"))
        
    iris_pred = int(GB_model.predict(input_features))
    
    prediction = (iris_pred, iris_types[iris_pred])
    print('\nPrediction with GB model:\n',prediction)

    probabil = GB_model.predict_proba(input_features) 
    probab_setosa = round(float(probabil[:,0]),6)
    probab_versicolor = round(float(probabil[:,1]),6)
    probab_virginica = round(float(probabil[:,2]),6)
          
    print('\nProbabilities with GB model:',
          '\nprob. setosa:',probab_setosa,
          '\nprob. versicolor:',probab_versicolor,
          '\nprob. virginica:',probab_virginica)
    
    exec_time = time.time() - start_time
    print('\nExecution time with GB model:\n',round(exec_time,6),'\n')
    
    return {'model_accuracy': accuracy,
            'test_support': support,
            'iris types': iris_types,
            'scores_setosa': score_GB.loc['0'],
            'scores_versicolor': score_GB.loc['1'],
            'scores_virginica': score_GB.loc['2'],
            'input_features:': input_features_dict,
            'prediction': prediction, 
            'probab_setosa': probab_setosa, 
            'probab_versicolor': probab_versicolor,
            'probab_virginica': probab_virginica,
            'exec_time': exec_time}