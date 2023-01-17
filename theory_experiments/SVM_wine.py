import torch
import numpy as np
from sklearn.datasets import load_wine
from datetime import datetime
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from src.parameters import Parameters
from src.dataset import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
import json

def SVM_wine(params):
    datetime_time = datetime.now().strftime("%d_%b_%Y (%H_%M_%S_%f)").replace(" ", "")
    params.d = 5
    file_name = "timestamp="+datetime_time+"_SVM_btrain="+str(params.b_train)+"_hiddensize="+str(params.hidden_size)
    random_seed = params.seed
    experiment_dict = {}
    torch.manual_seed(params.seed)
    np.random.seed(params.seed)
    initial_setting_dict = {"random_seed":params.seed}

    output_size = 3
    input_size=13
    valid_size = 0.25
    

    gammas = np.linspace(-11.51, -2.23, num=100)
    #gammas = [-2]
    Cs = np.linspace(-6.9, 4.6, num=100)
    #Cs = [0.001]
    initial_setting_dict['output_size']=output_size
    initial_setting_dict['input_size']=input_size
    experiment_dict['init_settings'] = initial_setting_dict
    accuracies = []
    valid_losses = []
    hyperparam_list = []
    for _C in Cs:
        for gamma in gammas:
            MNIST_experiment = {}
            hyperparams = np.array([gamma, _C])
            hyperparams = hyperparams.T
            dataset = Dataset(params)
            dataset.data.problem = "SVM"
            X, y = load_wine(return_X_y=True)
            print(X.shape)
            print(y.shape)
            train_X, valid_X, train_y, valid_y = train_test_split(
                X, y, test_size=valid_size, random_state=0
            )
            clf = make_pipeline(StandardScaler(), SVC(C=_C, gamma=np.exp(gamma)))
            clf.fit(train_X, train_y)

            preds = clf.predict(valid_X)
            accuracy = accuracy_score(valid_y, preds)
            accuracies.append(accuracy)

            #probs = clf.predict_proba(valid_X)
            #loss = log_loss(valid_y, probs)
            #valid_losses.append(loss)
            hyperparam_list.append(hyperparams.T.tolist())
    MNIST_results = {}
    MNIST_results["hyperparams"] = hyperparam_list
    #MNIST_results["valid_losses"] = valid_losses
    MNIST_results["accuracies"] = accuracies
    experiment_dict['MNIST_results'] = MNIST_results
#    with open("/zhome/49/2/135395/PhD/unibo/results/SVM_wine/" + file_name+ ".json", 'w') as fp:
#        json.dump(experiment_dict, fp, indent=4)
#    with open("./results/SVM_wine/" + file_name+ ".json", 'w') as fp:
#        json.dump(experiment_dict, fp, indent=4)
if __name__ == "__main__":
    SVM_wine()
