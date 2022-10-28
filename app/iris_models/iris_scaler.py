import pickle

def scale_inputs(input_features):
    
    scaler = pickle.load(open('iris_models//iris_scaler.pkl','rb'))
    
    input_features_scaled = scaler.transform(input_features)
    
    input_features_scaled_dict = {'sepal length': round(input_features_scaled[0][0],6), 
                                  'sepal  width': round(input_features_scaled[0][1],6), 
                                  'petal length': round(input_features_scaled[0][2],6), 
                                  'petal width': round(input_features_scaled[0][3],6)}
    
    print('\nInput features scaled\n', input_features_scaled_dict)
    
    return {'input_features_scaled': input_features_scaled,
            'input_features_scaled_dict': input_features_scaled_dict}