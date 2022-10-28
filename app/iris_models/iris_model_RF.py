import pickle
import time
import pandas as pd

def predict_RF(input_features, input_features_dict, iris_types):
    
    start_time = time.time()

    score_RF = pd.read_csv('iris_data//iris_report_RF.csv').set_index('class')
    
    accuracy = score_RF.loc['accuracy'].mean()
    support = score_RF.iloc[-1,-1]
    
    print('\nAccuracy of the RF model:\n', accuracy)
    print('\nTest Support of the RF model:\n', support)
    print('\nScore of the RF model:\n', score_RF.iloc[0:3,:])    
    
    RF_model = pickle.load(open("iris_models//iris_model_RF.pkl", "rb"))
        
    iris_pred = int(RF_model.predict(input_features))
    
    prediction = (iris_pred, iris_types[iris_pred])
    print('\nPrediction with RF model:\n',prediction)

    probabil = RF_model.predict_proba(input_features) 
    probab_setosa = round(float(probabil[:,0]),6)
    probab_versicolor = round(float(probabil[:,1]),6)
    probab_virginica = round(float(probabil[:,2]),6)
          
    print('\nProbabilities with RF model:',
          '\nprob. setosa:',probab_setosa,
          '\nprob. versicolor:',probab_versicolor,
          '\nprob. virginica:',probab_virginica)
    
    exec_time = time.time() - start_time
    print('\nExecution time with RF model:\n',round(exec_time,6),'\n')
    
    return {'model_accuracy': accuracy,
            'test_support': support,
            'iris types': iris_types,
            'scores_setosa': score_RF.loc['0'],
            'scores_versicolor': score_RF.loc['1'],
            'scores_virginica': score_RF.loc['2'],
            'input_features:': input_features_dict,
            'prediction': prediction, 
            'probab_setosa': probab_setosa, 
            'probab_versicolor': probab_versicolor,
            'probab_virginica': probab_virginica,
            'exec_time': exec_time}