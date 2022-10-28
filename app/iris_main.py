import json
from fastapi import FastAPI, Query
from pydantic import BaseModel
import pickle
import logging
from iris_models.iris_model_SVM import predict_SVM
from iris_models.iris_model_RF import predict_RF
from iris_models.iris_model_KM import predict_KM
from iris_models.iris_model_GB import predict_GB
from iris_models.iris_model_LR import predict_LR
from iris_models.iris_model_NN import predict_NN
import uvicorn

# Initialize logging

my_logger = logging.getLogger()
my_logger.setLevel(logging.DEBUG)
logging.basicConfig(level=logging.DEBUG, filename='iris_logs.log')

iris_types = {0: 'setosa', 
              1: 'versicolor', 
              2: 'virginica'}
    
app = FastAPI(title="Iris ML API", description="API for iris dataset, using different ml models", docs_url='/') 
    
@app.on_event('startup')
def startup():
    print ('\nTo make a prediction,\ninsert inputs here below (fastAPI):\nhttp://localhost:8000/#/Model_Inputs_Selection/model_selection_model_selection_get\nor here below (Docker):\nhttp://localhost:7000/#/Model_Inputs_Selection/model_selection_model_selection_get\n')
    return('testest')

# Model and input features selection

@app.get('/model_selection',  tags=["Model_Inputs_Selection"]) 
async def model_selection(ModelSelected: str = Query(enum=["Kmeans",
                                                           "LogReg",
                                                           "SupVecMac",
                                                           "GradBoost",
                                                           "RandFor",
                                                           "NeurNet"]),
                          SepalLength: float = 0.0,
                          SepalWidth: float = 0.0,
                          PetalLength: float = 0.0,
                          PetalWidth: float = 0.0):

    input_features = [[SepalLength, SepalWidth, PetalLength, PetalWidth]]
    input_features_dict = {'sepal length': SepalLength, 
                           'sepal  width': SepalWidth, 
                           'petal length': PetalLength, 
                           'petal width': PetalWidth}

    print('\nModel Selected:\n', ModelSelected)
    
    print ('\nInput features inserted:\n', input_features_dict)
    
    if ModelSelected == 'Kmeans':
        return (predict_KM(input_features,input_features_dict,iris_types))
    if ModelSelected == 'LogReg':
        return (predict_LR(input_features,input_features_dict,iris_types))
    if ModelSelected == 'SupVecMac':
        return (predict_SVM(input_features,input_features_dict,iris_types))
    if ModelSelected == 'RandFor':
        return (predict_RF(input_features,input_features_dict,iris_types))
    if ModelSelected == 'GradBoost':
        return (predict_GB(input_features,input_features_dict,iris_types))
    if ModelSelected == 'NeurNet':
        return (predict_NN(input_features,input_features_dict,iris_types))
    else:
        print ('\nModel *', ModelSelected, '* not yet implemented\n')
        return 'Model', ModelSelected, 'not yet implemented'

# if __name__ == "__main__":
#     uvicorn.run(app, host="127.0.0.1", port=8000)