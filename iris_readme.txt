**********************************************************
API for the models implemented in 'iris_models.ipynb'
**********************************************************
- open the terminal from the jupyter notebook
- locate the path of the 'iris_main.py'
- execute: 'uvicorn iris_main:app --reload'

**********************************************************
MLFlow
**********************************************************
- open the terminal
- locate the path of the 'mlruns' folder 
- execute: 'mlflow ui' 

**********************************************************
Docker
**********************************************************
- write requirements file
 pip freeze > iris_requirements.txt  <== all
 pipreqs C:\Users\Alienware\Documents\INGEGNERIA\DS\IrisProject\

- to remove container and image
 docker rm -f my_iris_container
 docker rmi -f alfrizz/iris_image

- Build the Docker image (execute it at the Dockerfile path level)
 docker build --no-cache . -t iris_image

- Run the Docker image as container
 docker run -dp 7000:7000 --name my_iris_container iris_image  

-push to repository
 docker login
 docker tag iris_image alfrizz/iris_image:latest
 docker push alfrizz/iris_image:latest

- to check the container filesystem
 docker exec -t -i my_iris_container /bin/bash

**********************************************************
Kubernetes
**********************************************************
New-Item -Path 'C:\Program Files' -Name 'minikube' -ItemType Directory -Force
Invoke-WebRequest -OutFile 'c:\Program Files\minikube\minikube.exe' -Uri 'https://github.com/kubernetes/minikube/releases/latest/download/minikube-windows-amd64.exe' -UseBasicParsing

$oldPath = [Environment]::GetEnvironmentVariable('Path', [EnvironmentVariableTarget]::Machine)
if ($oldPath.Split(';') -inotcontains 'C:\Program Files\minikube'){ `
  [Environment]::SetEnvironmentVariable('Path', $('{0};C:\Program Files\minikube' -f $oldPath), [EnvironmentVariableTarget]::Machine) `
}

curl -LO "https://dl.k8s.io/release/v1.25.0/bin/windows/amd64/kubectl.exe"

------

enable kubernetes in the docker desktop options

- to remove minikube container 
 docker rm -f minikube

- to start the minikube container
minikube start --driver=docker   

kubectl delete deployment iris-deploym
kubectl delete service iris-service

minikube tunnel

kubectl apply -f .......\deploym.yaml
kubectl apply -f .......\service.yaml

kubectl describe deployment iris-deploym
kubectl describe services iris-service

minikube service iris-service
minikube dashboard

kubectl get endpoints
kubectl cluster-info
kubectl get nodes --output=wide





