import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import  train_test_split
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score, roc_auc_score
import matplotlib.pyplot as plt
from pathlib import Path
import wandb
import pprint
import pickle
import os
import random
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class EEGNET(nn.Module):
    def __init__(self, receptive_field, filter_sizing, mean_pool, activation_type, dropout, D):
        super(EEGNET,self).__init__()
        sample_duration = 500
        channel_amount = 8
        num_classes = 3
        self.temporal=nn.Sequential(
            nn.Conv2d(1,filter_sizing,kernel_size=[1,receptive_field],stride=1, bias=False,\
                padding='same'), 
            nn.BatchNorm2d(filter_sizing),
        )
        self.spatial=nn.Sequential(
            nn.Conv2d(filter_sizing,filter_sizing*D,kernel_size=[channel_amount,1],bias=False,\
                groups=filter_sizing),
            nn.BatchNorm2d(filter_sizing*D),
            nn.ELU(True),
        )
        self.seperable=nn.Sequential(
            nn.Conv2d(filter_sizing*D,filter_sizing*D,kernel_size=[1,16],\
                padding='same',groups=filter_sizing*D, bias=False),
            nn.Conv2d(filter_sizing*D,filter_sizing*D,kernel_size=[1,1], padding='same',groups=1, bias=False),
            nn.BatchNorm2d(filter_sizing*D),
            nn.ELU(True),
        )
        self.avgpool1 = nn.AvgPool2d([1, 5], stride=[1, 5], padding=0)   
        self.avgpool2 = nn.AvgPool2d([1, 5], stride=[1, 5], padding=0)
        self.dropout = nn.Dropout(dropout)
        self.view = nn.Sequential(Flatten())

        endsize = 320#filter_sizing*D*15
        self.fc2 = nn.Linear(endsize, num_classes)

    def forward(self,x):
        out = self.temporal(x)
        out = self.spatial(out)
        out = self.avgpool1(out)
        out = self.dropout(out)
        out = self.seperable(out)
        out = self.avgpool2(out)
        out = self.dropout(out)
        out = out.view(out.size(0), -1)
        prediction = self.fc2(out)
        return prediction

def data_setup(batch_size, val_subjects, test_subject):
    print(test_subject)
    result_path = Path(f"./results/intermediate_datafiles/openloopTL/TL_pretrain_for_{test_subject}")
    result_path.mkdir(exist_ok=True, parents=True)
    alldata_path = Path(f'./data/openloop/intermediate_datafiles/preprocess/TL_1_100Hz_onlygood')
    X_train, y_train, X_val, y_val, X_test, y_test = [], [], [], [], [], []
    for instance in os.scandir(alldata_path):
        print(f'Adding data for {instance.path}...')
        a_file = open(instance.path, "rb")
        data_dict = pickle.load(a_file)
        X = data_dict['data']
        y = data_dict['labels']
        for df in X:      
            for segment in range(len(X[df])): 
                # upperlimb classification
                if y[df][segment] == 0 or y[df][segment] == 1 or y[df][segment] == 2:
                    if not test_subject in instance.path:
                        if val_subjects[0] in instance.path:
                            #print(val_subjects[0])
                            if df < 4:
                                # put trials of unseen subject
                                X_val.append(X[df][segment])
                                y_val.append(y[df][segment]) 
                            else:
                                X_test.append(X[df][segment])
                                y_test.append(y[df][segment]) 
                        else:
                            X_train.append(X[df][segment])
                            y_train.append(y[df][segment])      
        print(f'Current length of X train: {len(X_train)}.')
        print(f'Current length of X val: {len(X_val)}.')
        print(f'Current length of X test: {len(X_test)}.')
    X_train_np = np.stack(X_train)
    X_val_np = np.stack(X_val)
    y_train_np = np.array(y_train)
    y_val_np = np.array(y_val)
    X_test_np = np.stack(X_test)
    y_test_np = np.array(y_test)

    trainX = torch.from_numpy(X_train_np)
    trainY = torch.from_numpy(y_train_np)
    validationX = torch.from_numpy(X_val_np)
    validationY = torch.from_numpy(y_val_np)
    testX = torch.from_numpy(X_test_np)
    testY = torch.from_numpy(y_test_np)

    train = torch.utils.data.TensorDataset(trainX, trainY)
    validation = torch.utils.data.TensorDataset(validationX, validationY)
    test = torch.utils.data.TensorDataset(testX, testY)
    trainloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    valloader = torch.utils.data.DataLoader(validation, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=True)
    return trainloader, valloader, testloader

def run():
    for subj in range(1,2):
        test_subject = f'X0{subj}'
        other_subjects = [2,4,5]#[1,2,3,4,5,6,7,8,9]
        #other_subjects.remove(subj)
        print(other_subjects)
        random.seed(subj)

        for i in other_subjects:
            val_subjects = [f"X0{i}"]
            config={
            'batch_size' : 256,
            'epochs': 30,
            'receptive_field': 64, 
            'mean_pool':  8,
            'activation_type':  'elu',
            'network' : 'EEGNET',
            'val_subjects': val_subjects,
            'test_subject': test_subject,
            'seed':  42,    
            'learning_rate': 0.001,
            'filter_sizing':  8,
            'D':  2,
            'dropout': 0.25}
            train(config)

def train(config=None):
    # Initialize a new wandb run
    with wandb.init(project=f"onlygoodsubj_EEGNET_v2-ft_PreTrain_for_{config['test_subject']}",config=config):
        config = wandb.config
        pprint.pprint(config)
        early_stopping = EarlyStopping()
        trainloader, valloader, testloader = data_setup(config.batch_size, config.val_subjects, config.test_subject)
        net = build_network(config)
        wandb.watch(net, log_freq=100)
        optimizer = optim.Adam(net.parameters(), lr=config.learning_rate)
        for epoch in range(config.epochs):
            print(f"Epoch num: {epoch}")
            train_loss, train_acc, train_f1 = train_epoch(net, trainloader, optimizer)
            #first_params = net.parameters
            val_loss, val_acc, val_f1, ft_acc = evaluate(net, valloader, testloader, optimizer)
            #print(f"This should be True: {net.parameters == first_params}")
            wandb.log({"epoch": epoch,
            "train/train_loss": train_loss,
            "train/train_acc": train_acc,
            "train/train_f1": train_f1,
            "val/val_loss": val_loss,
            "val/vaL_acc": val_acc,
            "val/val_f1": val_f1,
            "val_finetune_acc": ft_acc}) 

            early_stopping(val_loss)
            if early_stopping.early_stop:
                break
        torch.save(net.state_dict(), f'pretrain_models/{config.test_subject}/EEGNET_ft_v2_onlygood/EEGNET-PreTrain_val{config.val_subjects[0]}')

def build_network(config):
    if config.network == 'EEGNET':
        net = EEGNET(config.receptive_field, config.filter_sizing, config.mean_pool, config.activation_type, config.dropout, config.D)
    pytorch_total_params_train = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f'trainable parameters: {pytorch_total_params_train}')
    return net.to(device)

def train_epoch(net, loader, optimizer):
    acc, running_loss, f1, batches, total = 0, 0, 0, 0, 0
    for _, (data, target) in enumerate(loader):
        data = data[:, np.newaxis, :, :]
        data, target = data.to(device, dtype=torch.float), target.to(device, dtype=torch.long)
        optimizer.zero_grad()
        # ➡ Forward pass
        loss = F.cross_entropy(net(data), target)
        running_loss += loss.item()
        _, predicted = torch.max(net(data).data, 1)
        acc += (predicted == target).sum().item()
        f1 += f1_score(target.data, predicted, average='macro')
        # ⬅ Backward pass + weight update
        loss.backward()
        optimizer.step()
        wandb.log({"batch loss": loss.item()})
        batches += 1 
        total += target.size(0)
    running_loss = running_loss / len(loader)
    acc =  acc / total
    f1 =  f1 / batches
    return running_loss, acc, f1

def evaluate(net, valloader, testloader, optimizer):
    acc, running_loss, f1, batches, total = 0, 0, 0, 0, 0
    current_net = copy.deepcopy(net)
    current_opt = copy.deepcopy(optimizer)
    for epoch in range(5):
        valacc, valrunning_loss, valf1, valbatches, valtotal = 0, 0, 0, 0, 0
        for _, (data, target) in enumerate(valloader):
            data = data[:, np.newaxis, :, :]
            data, target = data.to(device, dtype=torch.float), target.to(device, dtype=torch.long)
            optimizer.zero_grad()
            loss = F.cross_entropy(current_net(data), target)
            valrunning_loss += loss.item()
            _, predicted = torch.max(current_net(data).data, 1)
            valacc += (predicted == target).sum().item()
            valf1 += f1_score(target.data, predicted, average='macro')
            loss.backward()
            current_opt.step()
            valbatches += 1
            valtotal += target.size(0)
        print(f'valacc epoch {epoch}: {valacc / valtotal}')
    #print(f"This should be false: {current_net.parameters() == net.parameters()}")
    #print(f"This should be false: {current_opt == optimizer}")

    for _, (data, target) in enumerate(testloader):
        data = data[:, np.newaxis, :, :]
        data, target = data.to(device, dtype=torch.float), target.to(device, dtype=torch.long)
        output = current_net(data)
        loss = F.cross_entropy(output, target)
        running_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        acc += (predicted == target).sum().item()
        f1 += f1_score(target.data, predicted, average='macro')
        batches += 1
        total += target.size(0)

    running_loss = running_loss / len(testloader)
    acc =  acc / total
    f1 =  f1 / batches
    print(f"test acc: {acc}, test f1: {f1}")
    ft_acc = valacc / valtotal

    return running_loss, acc, f1, ft_acc

class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=5, min_delta=1e-8):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True


