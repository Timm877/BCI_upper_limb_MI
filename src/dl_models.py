       
class CNN(nn.Module):
    def __init__(self, sample_duration, channel_amount, receptive_field, filter_sizing, mean_pool, num_classes):
        super(CNN,self).__init__()
        self.temporal=nn.Sequential(
            nn.Conv2d(1,filter_sizing,kernel_size=[1,receptive_field],stride=1, padding=0), 
            nn.BatchNorm2d(filter_sizing),
            nn.ELU(True),
        )
        self.spatial=nn.Sequential(
            nn.Conv2d(filter_sizing,filter_sizing,kernel_size=[channel_amount,1],padding=0),
            nn.BatchNorm2d(filter_sizing),
            nn.ELU(True),
        )
        self.avgpool = nn.AvgPool2d([1, mean_pool], stride=[1, mean_pool], padding=0)
        self.dropout = nn.Dropout(0.25)
        self.view = nn.Sequential(Flatten())

        endsize = filter_sizing*((sample_duration-receptive_field+1)//mean_pool)
        print(endsize)
        #self.fc1= nn.Sequential(nn.Linear(endsize, 100), nn.ReLU())
        self.fc2= nn.Linear(endsize, num_classes)

    def forward(self,x):
        out = self.temporal(x)
        out = self.dropout(out)
        out = self.spatial(out)
        out = self.dropout(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        #out = self.fc1(out)
        prediction = self.fc2(out)
        return prediction
       
class CNN(nn.Module):
    def __init__(self, sample_duration, channel_amount, receptive_field, filter_sizing, mean_pool, num_classes):
        super(CNN,self).__init__()
        self.temporal=nn.Sequential(
            nn.Conv2d(1,filter_sizing,kernel_size=[1,receptive_field],stride=1, padding=0), 
            nn.BatchNorm2d(filter_sizing),
            nn.ELU(True),
        )
        self.spatial=nn.Sequential(
            nn.Conv2d(filter_sizing,filter_sizing,kernel_size=[channel_amount,1],padding=0),
            nn.BatchNorm2d(filter_sizing),
            nn.ELU(True),
        )
        self.avgpool = nn.AvgPool2d([1, mean_pool], stride=[1, mean_pool], padding=0)
        self.dropout = nn.Dropout(0.25)
        self.view = nn.Sequential(Flatten())

        endsize = filter_sizing*((sample_duration-receptive_field+1)//mean_pool)
        print(endsize)
        #self.fc1= nn.Sequential(nn.Linear(endsize, 100), nn.ReLU())
        self.fc2= nn.Linear(endsize, num_classes)

    def forward(self,x):
        out = self.temporal(x)
        out = self.dropout(out)
        out = self.spatial(out)
        out = self.dropout(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        #out = self.fc1(out)
        prediction = self.fc2(out)
        return prediction


class Inception(nn.Module):
    def __init__(self, receptive_field, filter_sizing, mean_pool, activation_type, dropout, D):
        super(Inception,self).__init__()
        sample_duration = 500
        channel_amount = 8
        num_classes = 3
        D=1
        receptive_field = [16, 32, 64]
        filter_sizing = [2, 4, 8]
        total_filtersize = (2+4+8)*D
        # block 1
        self.temporal1=nn.Sequential(
            nn.Conv2d(1,filter_sizing[0],kernel_size=[1,receptive_field[0]],stride=1, bias=False,\
                padding='same'), 
            nn.BatchNorm2d(filter_sizing[0]),
        )
        self.spatial1=nn.Sequential(
            nn.Conv2d(filter_sizing[0],filter_sizing[0]*D,kernel_size=[channel_amount,1],bias=False,\
                groups=filter_sizing[0]),
            nn.BatchNorm2d(filter_sizing[0]*D),
            nn.ELU(True),
        )
        # ----------------------

        
        # block 2
        self.temporal2 = nn.Sequential(
            nn.Conv2d(1,filter_sizing[1],kernel_size=[1,receptive_field[1]],stride=1, bias=False,\
                padding='same'), 
            nn.BatchNorm2d(filter_sizing[1]),
        )
        self.spatial2 = nn.Sequential(
            nn.Conv2d(filter_sizing[1],filter_sizing[1]*D,kernel_size=[channel_amount,1],bias=False,\
                groups=filter_sizing[1]),
            nn.BatchNorm2d(filter_sizing[1]*D),
            nn.ELU(True),
        )
        # ----------------------

        # block 3
        self.temporal3 = nn.Sequential(
            nn.Conv2d(1,filter_sizing[2],kernel_size=[1,receptive_field[2]],stride=1, bias=False,\
                padding='same'), 
            nn.BatchNorm2d(filter_sizing[2]),
        )
        self.spatial3 = nn.Sequential(
            nn.Conv2d(filter_sizing[2],filter_sizing[2]*D,kernel_size=[channel_amount,1],bias=False,\
                groups=filter_sizing[2]),
            nn.BatchNorm2d(filter_sizing[2]*D),
            nn.ELU(True),
        )
        # ----------------------
        self.TC1_1 = nn.Sequential(
            nn.Conv2d(total_filtersize, total_filtersize, kernel_size=[1,4], stride=1, bias=False,\
                padding='valid', dilation=(1,1)), 
            nn.BatchNorm2d(total_filtersize),
            nn.ELU(True),
        )
        self.TC1_2 = nn.Sequential(
            nn.Conv2d(total_filtersize, total_filtersize,kernel_size=[1,4], stride=1, bias=False,\
                padding='valid', dilation=(1,1)), 
            nn.BatchNorm2d(total_filtersize),
            nn.ELU(True),
        )
        # ----------------------
        self.TC2_1 = nn.Sequential(
            nn.Conv2d(total_filtersize, total_filtersize, kernel_size=[1,4], stride=1, bias=False,\
                padding='valid', dilation=(1,2)), 
            nn.BatchNorm2d(total_filtersize),
            nn.ELU(True),
        )
        self.TC2_2 = nn.Sequential(
            nn.Conv2d(total_filtersize, total_filtersize,kernel_size=[1,4], stride=1, bias=False,\
                padding='valid', dilation=(1,2)), 
            nn.BatchNorm2d(total_filtersize),
            nn.ELU(True),
        )
        # ----------------------
        self.TC3_1 = nn.Sequential(
            nn.Conv2d(total_filtersize, total_filtersize, kernel_size=[1,4], stride=1, bias=False,\
                padding='valid', dilation=(1,4)), 
            nn.BatchNorm2d(total_filtersize),
            nn.ELU(True),
        )
        self.TC3_2 = nn.Sequential(
            nn.Conv2d(total_filtersize, total_filtersize,kernel_size=[1,4], stride=1, bias=False,\
                padding='valid', dilation=(1,4)), 
            nn.BatchNorm2d(total_filtersize),
            nn.ELU(True),
        )

        # ----------------------
        self.Conv1x1 = nn.Sequential(
            nn.Conv2d(total_filtersize, total_filtersize,kernel_size=[1,1], stride=1, bias=False,\
                padding='valid'), 
            nn.BatchNorm2d(total_filtersize),
            nn.ELU(True),
        )

        self.avgpool1 = nn.AvgPool2d([1, 5], stride=[1, 5], padding=0)  
        #self.avgpool1 = nn.AvgPool2d([1, 5], stride=[1, 5], padding=0) 

        #self.avgpool2 = nn.AvgPool3d([4, 1, 1], stride=[4, 1, 1], padding=0)
        self.dropout = nn.Dropout(dropout)
        self.view = nn.Sequential(Flatten())

        self.seperable=nn.Sequential(
            nn.Conv2d(total_filtersize,total_filtersize,kernel_size=[1,16],\
                padding='same',groups=total_filtersize, bias=False),
            nn.Conv2d(total_filtersize,total_filtersize,kernel_size=[1,1], padding='same',groups=1, bias=False),
            nn.BatchNorm2d(total_filtersize),
            nn.ELU(True),
        )

        endsize = 280
        self.fc = nn.Linear(endsize, num_classes)

    def forward(self,x):
        block1 = self.spatial1(self.temporal1(x))
        #print(block1.size())
        block2 = self.spatial2(self.temporal2(x))
        #print(block2.size())
        block3 = self.spatial3(self.temporal3(x))
        #print(block3.size())
        block = torch.cat([block1, block2, block3], axis=1)
        #print(block.size())
        block = self.avgpool1(block)
        block = self.dropout(block)
        out = self.avgpool1(self.seperable(block))
        
        #print(block_in.size())
        '''
        # ---------------------------
        block = F.pad(block, (3,0), "constant", 3)
        block = self.dropout(self.TC1_1(block))
        #print(block.size())
        block = F.pad(block, (3,0), "constant", 3)
        block = self.dropout(self.TC1_2(block))
        #print(block.size())
        block_out = torch.add(block_in, block)
        #print(block_out.size())
        # ---------------------------

        block = F.pad(block_out, (6,0), "constant", 3)
        block = self.dropout(self.TC2_1(block))
        #print(block.size())
        block = F.pad(block, (6,0), "constant", 3)
        block = self.dropout(self.TC2_2(block))
        #print(block.size())
        block_out = torch.add(block_out, block)
        #print(block_out.size())

        # ---------------------------

        block = F.pad(block_out, (12,0), "constant", 3)
        block = self.dropout(self.TC3_1(block))
        #print(block.size())
        block = F.pad(block, (12,0), "constant", 3)
        block = self.dropout(self.TC3_2(block))
        #print(block.size())
        block_out = torch.add(block_out, block)
        #print(block_out.size())
        # ----------------
        '''
        #block = self.dropout(self.avgpool2(self.Conv1x1(block_out)))
        #print(block.size())
        out = out.view(out.size(0), -1)
        #print(out.size())
        prediction = self.fc(out)
        return prediction