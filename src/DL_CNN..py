# Rebuild the CNN approach of paper1 https://github.com/ByeonghooLee-ku/ICASSP2020-2020-code
# although I look more here https://arxiv.org/pdf/2002.01122v1.pdf which is an older paper of them
# paper2 = Deep Learning-Based Classification of Fine Hand Movements from Low Frequency EEG  https://www.mdpi.com/1999-5903/13/5/103/html
import torch.nn as nn

input_size = 250 #1 sec with 250Hz
receptive_field = 26 #In paper1, they use 65, but also collect more samples (3seconds)
channels = 27 # or 8 --> they had 24
filters = 40 #adapted from paper2
mean_pool = 15 #paper1 they use avg pool with stride 3, in paper2 they used 15
Shared_output = 600 #calculated for my case as 40x15 but might be better to not hard code this
Final_output = 2 # first, binary task.

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Shared(nn.Module):
    # init size: 27 channels x 250 samples 
    def __init__(self,outputSize):
        self.outputSize=outputSize
        super(Shared,self).__init__()
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
        # here data is 1 channel x 226 samples x 40 filters
        # they have here size of 210x36 which they pool w 15 making it 14x36 
        # this they than flatten to 14x36 = 504
        # btw they also apply a 2nd conv layer which they claim gives a better results

        # if my data here would be 225 samples it would be perfect for 15 size avg pool
        # thus let's change the receptive_field size to 26
        # now I end up with 40x15 after pooling
        self.avgpool = nn.AvgPool2d([1, mean_pool], stride=[1, mean_pool], padding=0)
    	# now data has become 40x15
        self.view = nn.Sequential(Flatten())
        # flattening gives 40x15=600
        self.fc=nn.Linear(Shared_output, Final_output)
        # final soutput is 2: binary task
    
    def forward(self,x):
        #x is 1 segment of channels x samples: 27x250
        out = self.temporal(x)
        out = self.spatial(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        prediction = self.fc(out)
        return prediction