# Train a classifier on the MedMNIST data
########################################################################################################################
# Import dependencies
import numpy as np
from models.classifier import  Encoder, Regressor
from datasets.numpy_dataset import numpy_dataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from utils import Args, EarlyStopping_split_models
import sys
import json
import torch.optim as optim
from torch.autograd import Variable
from losses.bhattacharya import bhattachayra_GMM
from GMM.gmm_EM import GaussianMixture

LOAD_PATH_ENCODER = 'pretrained_model/weight_encoder_site_1_checkpoint'
LOAD_PATH_REGRESSOR = 'pretrained_model/weight_regressor_site_1_checkpoint'

site = 2
components = 2

PATH_ENCODER = 'weight_encoder_site_' + str(site) + '_c_' + str(components)
CHK_PATH_ENCODER = 'weight_encoder_site_' + str(site) + '_c' + str(components) + '_checkpoint'
PATH_REGRESSOR = 'lweight_regressor_site_' + str(site) + '_c_' + str(components)
CHK_PATH_REGRESSOR = 'weight_regressor_site_' + str(site) + '_c' + str(components) + '_checkpoint'
LOSS_PATH = 'losses_site_' + str(site) + '_c_' + str(components)

ref_mu = np.load('pretrained_model/normal_model_mednist_X1_training_X1_mu_' + str(components)  +'.npy')
ref_sigma = np.sqrt(np.load('pretrained_model/normal_model_mednist_X1_training_X1_var_' + str(components)  +'.npy'))
ref_pi = np.load('pretrained_model/normal_model_mednist_X1_training_X1_pi_' + str(components)  +'.npy')
print('Reference Distributions: ')
print('Mu: ', ref_mu.shape)
print('Sigma: ', ref_sigma.shape)
print('Pi: ', ref_pi.shape)

weight = 1
# Create an args class
args = Args()
args.channels_first = True
args.epochs = 500
args.batch_size = 5
args.alpha = weight
args.patience = 10
args.train_val_prop = 0.75
args.learning_rate = 1e-6

cuda = torch.cuda.is_available()
########################################################################################################################
def train_bhatt_GMM(args, models, train_loader, optimizer, criterion, epoch):
    [encoder, regressor] = models
    
    bhatt_loss = 0
    
    encoder.train()
    regressor.eval()
    gmm_model = GaussianMixture(32, components, 1, covariance_type="diag", init_params="random")
    batches = 0
    for batch_idx, (total_data, _) in enumerate(train_loader):
        total_data =  Variable(total_data)
        
        if list(total_data.size())[0] == args.batch_size :
            batches += 1
            # First update the encoder and regressor
            optimizer.zero_grad()
            features = encoder(total_data)
            features = features.view(-1, 32, 1)

            gmm_model.fit(features, n_iter=20)
                        
            loss = criterion(gmm_model.mu, torch.sqrt(gmm_model.var), gmm_model.pi)
            loss.backward()
            optimizer.step()
            
            bhatt_loss += loss.detach().cpu().numpy()

            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, (batch_idx+1) * len(total_data), len(train_loader.dataset),
                           100. * (batch_idx+1) / len(train_loader), loss.item()), flush=True)

    av_bhatt = bhatt_loss / batches 
    print('Training set: Average Bhatt: {:.4f}'.format(av_bhatt,  flush=True))

    return  av_bhatt
def val_normal_bhatt(args, models, val_loader, criterion):
    [encoder, regressor] = models
    
    encoder.eval()
    regressor.eval()
    gmm_model = GaussianMixture(32, components, 1, covariance_type="diag", init_params="random")

    total_loss = 0
    batches = 0
    with torch.no_grad():
        for _,  (total_data, _) in enumerate( val_loader):
            total_data = Variable(total_data)

            if list(total_data.size())[0] == args.batch_size:
                batches += 1
                features = encoder(total_data)
                features = features.view(-1, 32, 1)
                gmm_model.fit(features, n_iter=20)
                loss = criterion(gmm_model.mu, torch.sqrt(gmm_model.var), gmm_model.pi)
                total_loss  += loss.detach().cpu().numpy()

    av_loss = total_loss / batches

    print('\nValidation set: Average loss: {:.4f}'.format(av_loss,  flush=True))
    print('\n') 
    return av_loss

#######################################################################################################################
X_total = np.load('data/X_' + str(site) + '_train.npy')
print(X_total.shape)

proportion = int(args.train_val_prop * len(X_total))
X_total_train = X_total[:proportion]
X_total_val = X_total[proportion:]
X_total_train, X_total_val = np.reshape(X_total_train, (-1, 1, 28, 28)), np.reshape(X_total_val, (-1, 1, 28, 28))

print('Training: ',X_total_train.shape, flush=True)
print('Validation: ', X_total_val.shape, flush=True)

total_train_dataset = numpy_dataset(X_total_train, X_total_train)
total_val_dataset = numpy_dataset(X_total_val, X_total_val)

total_train_dataloader = DataLoader(total_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
total_val_dataloader = DataLoader(total_val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

# Load the model
encoder = Encoder()
regressor = Regressor()

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

print('Freeze regressor')
for param in regressor.parameters():
    param.requires_grad = False
models = [encoder, regressor]

bhatt_criterion = bhattachayra_GMM(components, ref_mu, ref_sigma, ref_pi)
optimizer = optim.AdamW(list(encoder.parameters()), lr=args.learning_rate)

# Initalise the early stopping
early_stopping = EarlyStopping_split_models(args.patience, verbose=False)
loss_store = []

epoch_reached = 1
for epoch in range(epoch_reached, args.epochs+1):   
    train_loss = train_bhatt_GMM(args, models, total_train_dataloader, optimizer, bhatt_criterion, epoch)
    val_loss = val_normal_bhatt(args, models, total_val_dataloader, bhatt_criterion)

    loss_store.append([train_loss, val_loss])
    np.save(LOSS_PATH, np.array(loss_store))
    
    # Decide whether the model should stop training or not
    early_stopping(val_loss, models, epoch, optimizer, train_loss, [CHK_PATH_ENCODER, CHK_PATH_REGRESSOR])
    if early_stopping.early_stop:
        loss_store = np.array(loss_store)
        np.save(LOSS_PATH, loss_store)
        sys.exit('Patience Reached - Early Stopping Activated')

    if epoch == args.epochs:
        print('Finished Training', flush=True)
        print('Saving the model', flush=True)

        # Save the model in such a way that we can continue training later
        torch.save(encoder.state_dict(), PATH_ENCODER)
        torch.save(regressor.state_dict(), PATH_REGRESSOR)

        loss_store = np.array(loss_store)
        np.save(LOSS_PATH, loss_store)

    torch.cuda.empty_cache()  # Clear memory cache