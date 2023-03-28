# Predict file for mednist data
################################################################################ 
#Import dependencies
import numpy as np
from models.classifier import  Encoder2, Regressor2
from datasets.numpy_dataset import numpy_dataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from sklearn.utils import shuffle
from utils import Args, EarlyStopping_split_models
import sys
import json
import torch.optim as optim
from torch.autograd import Variable
from sklearn.metrics import accuracy_score
###############################################################################
def normal_predict(args, models, criterion, test_loader):
    age_results = []
    age_true = []
    [encoder, regressor] = models
    encoder.eval()
    regressor.eval()

    with torch.no_grad():
        for data, target in test_loader:
            data, target = Variable(data), Variable(target)
            if list(data.size())[0] == args.batch_size:
                age_true.append(target.detach().cpu().numpy())
                x = encoder(data)
                age_preds = regressor(x)
                pred = np.argmax(age_preds.detach().numpy(), axis=1)
                age_results.append(pred.reshape(-1))

    age_results = np.array(age_results)
    age_results_cp = np.copy(age_results)
    del age_results
    age_true = np.array(age_true)
    age_true_cp = np.copy(age_true)
    del age_true

    return age_results_cp, age_true_cp

def get_embeddings(args, models, criterion, test_loader):
    [encoder, regressor] = models
    encoder.eval()
    regressor.eval()
    emb_store = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = Variable(data), Variable(target)
            if list(data.size())[0] == args.batch_size:
                x = encoder(data)
                emb_store.append(x.detach().cpu().numpy())
    return np.array(emb_store)
########################################################################################################################
# Create an args class
args = Args()
args.channels_first = True
args.epochs = 25
args.batch_size = 1
args.diff_model_flag = False
args.alpha = 100
args.patience = 25
args.train_val_prop = 0.8
args.learning_rate = 1e-5

components = 2
########################################################################################################################
for i in [2, 3, 4, 5]:
    
    LOAD_PATH_ENCODER = 'weight_encoder_site_' + str(i) + '_c' + str(components) + '_checkpoint'
    LOAD_PATH_REGRESSOR = 'weight_regressor_site_' + str(i) + '_c' + str(components) + '_checkpoint'

    print('Domain ' + str(i))

    X = np.load('data/X_' + str(i) + '_test.npy')
    y = np.load('data/y_' + str(i) + '_test.npy')

    X = np.reshape(X, (-1, 1, 28, 28))
    print('Data splits')
    print(X.shape, y.shape)

    datatset = numpy_dataset(X, y)
    dataloader = DataLoader(datatset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Load the model
    encoder = Encoder2()
    regressor = Regressor2()

    if LOAD_PATH_ENCODER:
        print('Loading Weights')
        encoder_dict = encoder.state_dict()
        pretrained_dict = torch.load(LOAD_PATH_ENCODER)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in encoder_dict}
        print('weights loaded unet = ', len(pretrained_dict), '/', len(encoder_dict))
        encoder.load_state_dict(torch.load(LOAD_PATH_ENCODER))
    if LOAD_PATH_REGRESSOR:
        print('Loading Weights')
        encoder_dict = regressor.state_dict()
        pretrained_dict = torch.load(LOAD_PATH_REGRESSOR)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in encoder_dict}
        print('weights loaded segmenter = ', len(pretrained_dict), '/', len(encoder_dict))
        regressor.load_state_dict(torch.load(LOAD_PATH_REGRESSOR))

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(list(encoder.parameters()) + list(regressor.parameters()), lr=args.learning_rate)
    early_stopping = EarlyStopping_split_models(args.patience, verbose=False)
    epoch_reached = 1
    loss_store = []

    models = [encoder, regressor]

    print('Predicting ' + str(i))
    results, true = normal_predict(args, models, criterion, dataloader)
    print(results.shape)
    np.save('bhatt_2c_X_' + str(i) + '_test_pred', results)
    np.save('bhatt_2c_X_'+ str(i) + '_test_true', true)


