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
receptive_field = 21 #Laura's data: use 21, us: use 26. In paper1, they use 65, but also collect more samples (3seconds)
channels = 27 # or 8 --> they had 24
filters = 20 #depth
mean_pool = 15 #paper1 they use avg pool with stride 3, in paper2 they used 15
Shared_output = 600 #calculated for my case as 40x15 but might be better to not hard code this
classes = 2 # first, binary task.

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class CNN(nn.Module):
    # init size Laura: 27x200, us: 8 channels x 250 samples 
    def __init__(self):
        super(CNN,self).__init__()
        self.temporal=nn.Sequential(
            nn.Conv2d(1,filters,kernel_size=[1,receptive_field],stride=1, padding=0), 
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
        # TODO assert shape 
        # here data is 1 channel x 226 samples x 40 filters
        # they have here size of 210x36 which they pool w 15 making it 14x36 
        # this they than flatten to 14x36 = 504
        # btw they also apply a 2nd conv layer which they claim gives a better results

        # if my data here would be 225 samples it would be perfect for 15 size avg pool
        # thus let's change the receptive_field size to 26
        # now I end up with 40x15 after pooling
        self.avgpool = nn.AvgPool2d([1, mean_pool], stride=[1, mean_pool], padding=0)

        # maybe experiment with another conv layer 
        #self.conv1=nn.Sequential(
        #    nn.Conv2d(filters,filters,kernel_size=[1,receptive_field],padding=0),
        #    nn.BatchNorm2d(36),
        #    nn.ELU(True)
        #)
        #self.avgpool2=nn.AvgPool2d(kernel_size=[1,mean_pool],stride=[1,mean_pool])

    	# now data has become 40x15
        self.view = nn.Sequential(Flatten())
        # flattening gives 40x15=600
        #TODO dont hardcode sizes of last line below but get with torch.shape o.i.d.
        self.fc=nn.Linear(20*12, classes)
        # final output is 2: binary task
    
    def forward(self,x):
        # TODO use assert instead of print
        #x is 1 segment of channels x samples: 27x200
        #print(f'shape x at start. Expected 1x27x200. True: {x.shape}')
        out = self.temporal(x)
        #print(f'shape x after temporal conv. Receptive field: {receptive_field}. Expected 40x27x180. True: {out.shape}')
        out = self.spatial(out)
        #print(f'shape x after spatial conv. Expected 40x1x180. True: {out.shape}')
        out = self.avgpool(out)
        #print(f'shape x after avgpool. Expected 40x1x12. True: {out.shape}')
        out = out.view(out.size(0), -1)
        #print(f'shape x after flattening. Expected {40*12}. True: {out.shape}')
        prediction = self.fc(out)
        #print(f'shape x after linear+softmax. Expected 2. True: {prediction.shape}')
        return prediction

def data_setup(X_np, y_np, val_size=0.2):
    X_train, X_val, y_train, y_val = train_test_split(X_np, y_np, test_size=val_size, shuffle=True, random_state=42)

    trainX = torch.from_numpy(X_train)
    trainY = torch.from_numpy(y_train)
    validationX = torch.from_numpy(X_val)
    validationY = torch.from_numpy(y_val)

    train = torch.utils.data.TensorDataset(trainX, trainY)
    validation = torch.utils.data.TensorDataset(validationX, validationY)

    trainloader = torch.utils.data.DataLoader(train, batch_size=16, shuffle=True)
    valloader = torch.utils.data.DataLoader(validation, batch_size=16, shuffle=True)

    return trainloader, valloader

def init_network():
    net = CNN()
    # load earlier model params: net.load_state_dict(torch.load(PATH))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net_cuda = net.to(device)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    return net 

def run_model(trainloader, valloader):
    net = CNN()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    val_accuracy = []
    train_accuracy = []
    for epoch in range(30):  # loop over the dataset 50 times (50 epochs)
        running_loss = 0
        train_acc = 0
        for i, (inputs, labels) in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs = inputs[:, np.newaxis, :, :]
            #print(f'Shape of inputs in batches: {inputs.shape}')
            inputs, labels = inputs.to(device, dtype=torch.float), labels.to(device, dtype=torch.long)
            # zero the parameter gradients
            optimizer.zero_grad()

            output = net(inputs) 
            loss = F.cross_entropy(output,labels)
            _, predicted = torch.max(output.data, 1)

            total = labels.size(0)
            correct = (predicted == labels).sum().item()
            train_acc += (100 * correct / total)
                
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            #print details every x batches
            if i % 10 == 9:
                train_accuracy.append(train_acc / 10)
                print('[%d, %5d] trainloss: %.3f' %
                        (epoch + 1, i + 1, running_loss / 10))
                print(f'Current train acc: {train_accuracy[-1]}')
                train_acc = 0.0
                running_loss = 0.0

                correct_val = 0
                total_val = 0
                for i, (inputs_val, labels_val) in enumerate(valloader, 0):
                    inputs_val = inputs_val[:, np.newaxis, :, :]
                    #print(f'Shape of inputs in batches: {inputs.shape}')
                    inputs_val, labels_val = inputs_val.to(device, dtype=torch.float), labels_val.to(device, dtype=torch.long)
                    output_val = net(inputs_val)
                    _, predicted_val = torch.max(output_val.data, 1)
                    total_val += labels_val.size(0)
                    correct_val += (predicted_val == labels_val).sum().item()
                val_accuracy.append(round(100 * correct_val / total_val,3))
                print(f'Current val acc: {val_accuracy[-1]}')

    print('Finished Training')
    print(f'Final accuracy of the network on the training set: {train_accuracy[-1]}')
    print(f'Final accuracy of the network on the training set: {val_accuracy[-1]}')
    return train_accuracy, val_accuracy
