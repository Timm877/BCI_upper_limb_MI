import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import  train_test_split
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score, roc_auc_score
import matplotlib.pyplot as plt

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class EEGNET(nn.Module):
    def __init__(self, receptive_field, filter_sizing, mean_pool, activation_type, dropout):
        super(EEGNET,self).__init__()
        sample_duration = 500
        channel_amount = 8
        num_classes = 3
        if activation_type == 'relu':
            self.temporal=nn.Sequential(
                nn.Conv2d(1,filter_sizing,kernel_size=[1,receptive_field],stride=1, padding=0), 
                nn.BatchNorm2d(filter_sizing),
            )
            self.spatial=nn.Sequential(
                nn.Conv2d(filter_sizing,filter_sizing,kernel_size=[channel_amount,1],padding=0),
                nn.BatchNorm2d(filter_sizing),
                nn.ReLU(),
            )
            self.seperable=nn.Sequential(
                nn.Conv2d(filter_sizing*2,filter_sizing*2,kernel_size=[1,16],padding=0),
                nn.BatchNorm2d(filter_sizing),
                nn.ReLU(),
            )
        else:
            self.temporal=nn.Sequential(
                nn.Conv2d(1,filter_sizing,kernel_size=[1,receptive_field],stride=1, bias=False,\
                    padding=(0, receptive_field//2)), 
                nn.BatchNorm2d(filter_sizing),
            )
            self.spatial=nn.Sequential(
                nn.Conv2d(filter_sizing,filter_sizing*2,kernel_size=[channel_amount,1],bias=False,\
                    groups=filter_sizing, max_norm=1),
                nn.BatchNorm2d(filter_sizing*2),
                nn.ELU(True),
            )
            self.seperable=nn.Sequential(
                nn.Conv2d(filter_sizing*2,filter_sizing*2,kernel_size=[1,16],\
                    padding=(0, 8),groups=filter_sizing*2, bias=False),
                nn.Conv2d(filter_sizing*2,filter_sizing*2,kernel_size=[1,16], groups=1, bias=False),
                nn.BatchNorm2d(filter_sizing),
                nn.ELU(True),
            )
        self.avgpool1 = nn.AvgPool2d([1, 5], stride=[1, 5], padding=0)   
        self.avgpool2 = nn.AvgPool2d([1, 5], stride=[1, 5], padding=0)
        self.dropout = nn.Dropout(dropout)
        self.view = nn.Sequential(Flatten())

        endsize = 320
        self.fc2= nn.Linear(endsize, num_classes)

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
 
def data_setup(X_train, y_train, X_val, y_val):
    trainX = torch.from_numpy(X_train)
    trainY = torch.from_numpy(y_train)
    validationX = torch.from_numpy(X_val)
    validationY = torch.from_numpy(y_val)

    train = torch.utils.data.TensorDataset(trainX, trainY)
    validation = torch.utils.data.TensorDataset(validationX, validationY)

    trainloader = torch.utils.data.DataLoader(train, batch_size=128, shuffle=True)
    valloader = torch.utils.data.DataLoader(validation, batch_size=128, shuffle=True)
    return trainloader, valloader

def run_model(trainloader, valloader, lr, sample_duration, channel_amount, receptive_field, filter_sizing, 
mean_pool, num_classes, pipeline_type):
    train_accuracy_iters = []
    val_accuracy_iters = []
    train_f1_iters = []
    val_f1_iters = []
    val_classacc_iters = []
    train_classacc_iters = []
    training_precision_iters = []
    training_recall_iters = []
    validation_precision_iters = []
    validation_recall_iters = []
    validation_roc_auc_iters = []

    print(f'To get a bit more stable results we run DL network from scratch 3 times.')
    for iteration in range(1):
        print(f'Running iteration {iteration+1}...')

        if pipeline_type == 'deep':
            net = EEGNET(sample_duration=sample_duration, channel_amount=channel_amount, receptive_field=receptive_field, 
            filter_sizing = filter_sizing, mean_pool=mean_pool, num_classes=num_classes)

        pytorch_total_params = sum(p.numel() for p in net.parameters())
        pytorch_total_params_train = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print(pytorch_total_params)
        print(pytorch_total_params_train)
        for name, param in net.named_parameters():
            print(f"{name}: {param.numel()}")

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net = net.to(device)
        optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        lr_scheduler = LRScheduler(optimizer)
        early_stopping = EarlyStopping()

        scheduler_is_used = False
        val_accuracy = []
        train_accuracy = []
        training_loss = []
        validation_loss = []
        training_f1 = []
        validation_f1 = []
        training_precision = []
        training_recall = []
        validation_precision = []
        validation_recall = []
        validation_roc_auc = []
        training_roc_auc = []
        val_acc_classes = []
        train_acc_classes=[]

        for epoch in range(100):
            running_loss = 0
            train_acc = 0
            net.train()
            for i, (inputs, labels) in enumerate(trainloader, 0):
                #if pipeline_type != 'deep_1dcnn':
                inputs = inputs[:, np.newaxis, :, :]
                #print(inputs.shape)
                inputs, labels = inputs.to(device, dtype=torch.float), labels.to(device, dtype=torch.long)
                optimizer.zero_grad()
                output = net(inputs) 
                loss = F.cross_entropy(output,labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            net.eval()
            running_loss = 0        

            # Calculate train and val accuracy after each epoch
            train_acc, train_loss, train_f1_score, train_acc_class, train_precision, train_recall, train_roc_auc = \
            calculate_metrics(trainloader, device, net, num_classes, pipeline_type)

            train_accuracy.append(train_acc)
            training_loss.append(train_loss)
            training_f1.append(train_f1_score)
            train_acc_classes.append(train_acc_class)
            training_precision.append(train_precision)
            training_recall.append(train_recall)
            training_roc_auc.append(train_roc_auc)
            print(f'Current train acc after epoch {epoch+1}: {train_accuracy[-1]}')
            print(f'Current train loss after epoch {epoch+1}: {training_loss[-1]}')

            val_acc, val_loss, val_f1_score, val_acc_class, val_precision, val_recall, val_roc_auc = calculate_metrics(
                valloader, device, net, num_classes, pipeline_type)

            val_accuracy.append(val_acc)
            validation_loss.append(val_loss)
            validation_f1.append(val_f1_score)
            val_acc_classes.append(val_acc_class)
            validation_precision.append(val_precision)
            validation_recall.append(val_recall)
            validation_roc_auc.append(val_roc_auc)
            print(f'Current val acc after epoch {epoch+1}: {val_accuracy[-1]}')
            print(f'Current val loss after epoch {epoch+1}: {validation_loss[-1]}\n')
            
            # Apply LR scheduler halving twice with patience 4, and after that do early stopping with patience 4.
            if scheduler_is_used == False:
                lr_scheduler(val_loss)
                for param_group in optimizer.param_groups:
                    if param_group['lr'] <  0.00022 : # thus lr has been decreased 2x
                        scheduler_is_used = True
            else:
                early_stopping(val_loss)
                if early_stopping.early_stop:
                    break

        print('Finished Training')
        print(f'Final accuracy of the network on the training set: {train_accuracy[-1]}')
        print(f'Final accuracy of the network on the validation set: {val_accuracy[-1]}')

        train_accuracy_iters.append(train_accuracy[-1])
        val_accuracy_iters.append(val_accuracy[-1])
        train_f1_iters.append(training_f1[-1])
        val_f1_iters.append(validation_f1[-1])
        train_classacc_iters.append(train_acc_classes[-1])
        val_classacc_iters.append(val_acc_classes[-1])
        
        training_precision_iters.append(training_precision[-1])
        training_recall_iters.append(training_recall[-1])
        validation_precision_iters.append(validation_precision[-1])
        validation_recall_iters.append(validation_recall[-1])
        validation_roc_auc_iters.append(validation_roc_auc[-1])

    return train_accuracy_iters, val_accuracy_iters, train_f1_iters, val_f1_iters, train_classacc_iters, val_classacc_iters, \
    training_precision_iters, training_recall_iters, validation_precision_iters, validation_recall_iters, validation_roc_auc_iters

def calculate_metrics(loader, device, net, num_classes, pipeline_type):
    correct = 0
    total = 0
    running_loss = 0
    f1 = 0
    prec, rec, roc_auc = 0, 0, 0
    acc_classes = np.zeros(num_classes)
    batches = 0
    for i, (inputs, labels) in enumerate(loader, 0):
        inputs = inputs[:, np.newaxis, :, :]
        inputs, labels = inputs.to(device, dtype=torch.float), labels.to(device, dtype=torch.long)
        output = net(inputs)
        loss = F.cross_entropy(output,labels)
        running_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += labels.size(0)
        batches += 1
        correct += (predicted == labels).sum().item()
        f1 += f1_score(labels.data, predicted, average='macro') 
        prec += precision_score(labels.data, predicted, average='macro', zero_division = 0)
        rec += recall_score(labels.data, predicted, average='macro')
    correct =  correct / total
    prec =  prec / batches
    rec = rec / batches
    roc_auc =  roc_auc / batches
    f1 =  f1 / batches
    acc_classes =  acc_classes / total
    return correct, running_loss, f1, acc_classes, prec, rec, roc_auc

class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=5, min_delta=1e-4):
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


class LRScheduler():
    """
    Learning rate scheduler. If the validation loss does not decrease for the 
    given number of `patience` epochs, then the learning rate will decrease by
    by given `factor`.
    """
    def __init__(
        self, optimizer, patience=5, min_lr=1e-6, factor=0.5
    ):
        """
        new_lr = old_lr * factor

        :param optimizer: the optimizer we are using
        :param patience: how many epochs to wait before updating the lr
        :param min_lr: least lr value to reduce to while updating
        :param factor: factor by which the lr should be updated
        """
        self.optimizer = optimizer
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor

        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( 
                self.optimizer,
                mode='min',
                patience=self.patience,
                factor=self.factor,
                min_lr=self.min_lr,
                verbose=True
            )
        

    def __call__(self, val_loss):
        self.lr_scheduler.step(val_loss)
