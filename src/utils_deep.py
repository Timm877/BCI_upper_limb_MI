# Rebuild the CNN approach of paper1 https://github.com/ByeonghooLee-ku/ICASSP2020-2020-code
# although I look more here https://arxiv.org/pdf/2002.01122v1.pdf which is an older paper of them
# paper2 = Deep Learning-Based Classification of Fine Hand Movements from Low Frequency EEG  https://www.mdpi.com/1999-5903/13/5/103/html
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import  train_test_split

input_size = 200 #Laura: 200Hz 1 sec, us: 1 sec with 250Hz
receptive_field_1 = 36 # chosen randomly. In paper1, they use 65, but also collect more samples (3seconds)
receptive_field_2 = 22 # chosen randomly. In paper1, they use 65, but also collect more samples (3seconds)
channels = 27 #eeg channels
filters = 40 # Chosen randomly. Depth of conv layers
mean_pool = 15 # value of 15 was used in both papers
classes = 2 # first, binary task.

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class CNN(nn.Module):
    # init size Laura: 27x200, us: 8 channels x 250 samples 
    def __init__(self):
        super(CNN,self).__init__()
        self.temporal=nn.Sequential(
            nn.Conv2d(1,filters,kernel_size=[1,receptive_field_1],stride=1, padding=0), 
            nn.BatchNorm2d(filters),
            nn.ELU(True),
        )
        # due to this conv step data reduces from 250 samples to [(W−K+2P)/S]+1 = (250 - 25+0)/1 +1 = 226
        # with filter of 40, size here becomes 27 channels x 226 samples x 40 filters
        self.spatial=nn.Sequential(
            nn.Conv2d(filters,filters,kernel_size=[channels,1],padding=0),
            nn.BatchNorm2d(filters),
            nn.ELU(True),
        )
        #added a 3rd conv layer as done in paper1

        self.temporal2=nn.Sequential(
            nn.Conv2d(filters,filters,kernel_size=[1,receptive_field_2],padding=0),
            nn.BatchNorm2d(filters),
            nn.ELU(True)
        )

        # here data is 1 channel x 226 samples x 40 filters
        # they have here size of 210x36 which they pool w 15 making it 14x36 
        # this they than flatten to 14x36 = 504
        # if my data here would be 225 samples it would be perfect for 15 size avg pool
        # e.g. I would end up with 40x15 after pooling
        self.avgpool = nn.AvgPool2d([1, mean_pool], stride=[1, mean_pool], padding=0)
        self.dropout = nn.Dropout(0.3)

        self.view = nn.Sequential(Flatten())
        # flattening gives filtxlength column vect

        self.fc=nn.Linear(filters*((input_size-receptive_field_1+1)//mean_pool), classes)
        # final output is 2: binary task
    
    def forward(self,x):
        #print(f'shape x at start. Expected 1x27x200. True: {x.shape}')
        out = self.temporal(x)
        #print(f'shape x after temporal conv. Receptive field: {receptive_field}. Expected 40x27x180. True: {out.shape}')
        out = self.spatial(out)
        #out = self.temporal2(out)
        #print(f'shape x after spatial conv. Expected 40x1x180. True: {out.shape}')
        out = self.avgpool(out)
        out = self.dropout(out)
        #print(f'shape x after avgpool. Expected 40x1x12. True: {out.shape}')
        out = out.view(out.size(0), -1)
        #print(f'shape x after flattening. Expected {40*12}. True: {out.shape}')
        prediction = self.fc(out)
        #print(f'shape x after linear+softmax. Expected 2. True: {prediction.shape}')
        return prediction

def data_setup(X_np, y_np, val_size=0.2):
    X_train, X_val, y_train, y_val = train_test_split(X_np, y_np, test_size=val_size, shuffle=True, random_state=4)

    trainX = torch.from_numpy(X_train)
    trainY = torch.from_numpy(y_train)
    validationX = torch.from_numpy(X_val)
    validationY = torch.from_numpy(y_val)

    train = torch.utils.data.TensorDataset(trainX, trainY)
    validation = torch.utils.data.TensorDataset(validationX, validationY)

    trainloader = torch.utils.data.DataLoader(train, batch_size=32, shuffle=True)
    valloader = torch.utils.data.DataLoader(validation, batch_size=32, shuffle=True)
    return trainloader, valloader

def run_model(trainloader, valloader, lr):
    net = CNN()
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
    for epoch in range(50):
        running_loss = 0
        train_acc = 0
        net.train()
        for i, (inputs, labels) in enumerate(trainloader, 0):
            inputs = inputs[:, np.newaxis, :, :]
            inputs, labels = inputs.to(device, dtype=torch.float), labels.to(device, dtype=torch.long)

            optimizer.zero_grad()
            output = net(inputs) 
            loss = F.cross_entropy(output,labels)
            '''
            _, predicted = torch.max(output.data, 1)
            total = labels.size(0)
            correct = (predicted == labels).sum().item()
            train_acc += (100 * correct / total)
            '''  
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        net.eval()
        #print(f'running train loss epoch {epoch + 1}: {round(running_loss,2)}')
        running_loss = 0        
        # Calculate train and val accuracy after each epoch
        train_acc, train_loss = calculate_metrics(trainloader, device, net)
        train_accuracy.append(train_acc)
        training_loss.append(train_loss)
        print(f'Current train acc after epoch {epoch+1}: {train_accuracy[-1]}')
        print(f'Current train loss after epoch {epoch+1}: {training_loss[-1] / len(training_loss)}')

        val_acc, val_loss = calculate_metrics(valloader, device, net)
        val_accuracy.append(val_acc)
        validation_loss.append(val_loss)
        print('\n')
        print(f'Current val acc after epoch {epoch+1}: {val_accuracy[-1]}')
        print(f'Current val loss after epoch {epoch+1}: {validation_loss[-1] / len(validation_loss)}')
        
        # here I could very easily add early stopping  and LR scheduler code:
        # Idea here is to apply LR scheduler once, and after that do early stopping.
        if scheduler_is_used == False:
            lr_scheduler(val_loss)
            for param_group in optimizer.param_groups:
                #print(param_group['lr'])
                if param_group['lr'] - lr != 0: # thus lr has been decreased
                    scheduler_is_used = True
        else:
            early_stopping(val_loss)
            if early_stopping.early_stop:
                break

    print('Finished Training')
    print(f'Final accuracy of the network on the training set: {train_accuracy[-1]}')
    print(f'Final accuracy of the network on the validation set: {val_accuracy[-1]}')
    return train_accuracy, val_accuracy

def calculate_metrics(loader, device, net):
    correct = 0
    total = 0
    running_loss = 0
    for i, (inputs, labels) in enumerate(loader, 0):
        inputs = inputs[:, np.newaxis, :, :]
        inputs, labels = inputs.to(device, dtype=torch.float), labels.to(device, dtype=torch.long)
        output = net(inputs)
        loss = F.cross_entropy(output,labels)
        running_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    return round(100 * correct / total, 3), round(running_loss,3)


class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=5, min_delta=0):
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
