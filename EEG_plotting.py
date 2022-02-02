'''
Created by Victor Delvigne
ISIA Lab, Faculty of Engineering University of Mons, Mons (Belgium)
IMT Nord Europe, Villeneuve d'Ascq (France)
victor.delvigne@umons.ac.be
Source: TBD
Copyright (C) 2021 - UMons/IMT Nord Europe
This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.
This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.
You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
'''

import msvcrt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from pylsl import StreamInlet, resolve_stream

# duration is just an arbitrary num / seconds before plotting / giving size to the plot
duration = 4
# sampling freq is the standard freq which the Unicorn sends
sampling_frequency = 250
# down sampling ratio is used to make a more smooth plot by only getting a sample w/ freq of 25
down_sampling_ratio = 10
# choosen electrode is 1 --> only show 1 electrode I guess
choosen_electrode = 1

plt.style.use('dark_background')

def main():
    streams = resolve_stream()
    inlet = StreamInlet(streams[0])

    aborted = False

    i = 0
    j = 0
    t = np.zeros((int(duration*sampling_frequency/down_sampling_ratio)))
    y = np.zeros((int(duration*sampling_frequency/down_sampling_ratio)))


    '''
    x = np.zeros((int(duration*sampling_frequency/dsown_sampling_ratio), int(8)))
    fig = plt.figure()
    ax = []
    for axes in range(1,9):
        ax.append(fig.add_subplot(8,1,axes))
    '''    

    while not aborted:
        j = i//down_sampling_ratio
        sample, timestamp = inlet.pull_sample()

        if i%down_sampling_ratio==0:
            if j < int(duration*sampling_frequency/down_sampling_ratio):
                t[j] = timestamp
                y[j] = sample[choosen_electrode]
                # let's see if we can plot data from all 8 electrodes
                #x[j,:] = sample[:8]
            else:
                # np.roll used to shift all 1 to the left
                # First, in the if statement above, we fill an array to a specific length we want 
                # (before plotting as the plot has a specific size)
                # here that is duration*sf/dsr, and when that array is full,
                # we continue by shifting all points one to the left so the oldest point now is at the back
                # and then we replace that oldest point with the newest one to continue the plot while
                # not increasing plot size etc
                t = np.roll(t, -1)
                y = np.roll(y, -1)
                
                t[-1] = timestamp
                y[-1] = sample[choosen_electrode]

                
                #   let's see if we can plot data from all 8 electrodes
                '''
                x= np.roll(x, -1)
                
                x[-1,:] = sample[:8]
                


                for i in range(8):
                    # TODO fix this plot
                    ax[i].clear() 
                    #std = np.std(x[:,i])
                    #ax[i].axis([t[0]-1, t[1]+1, np.min(x[:,i])-std, np.max(x[:,i])+std])
                    ax[i].plot(t, x[:,i], 'o-')
                plt.pause(0.05)
                '''


                '''
                ax1.clear()
                ax2.clear()

                std = np.std(x[:,0])
                ax1.axis([t[0]-1, t[1]+1, np.min(x[:,0])-std, np.max(x[:,0])+std])
                ax1.plot(t, x[:,0], 'o-', c='c')

                std = np.std(x[:,1])
                ax2.axis([t[0]-1, t[1]+1, np.min(x[:,1])-std, np.max(x[:,1])+std])
                ax2.plot(t, x[:,1], 'o-', c='c')

                plt.pause(0.01)
                '''

               
                std = np.std(y)

                plt.axis([t[0]-1, t[1]+1, np.min(y)-std, np.max(y)+std])
                plt.plot(t, y, 'o-', c='c')
                plt.pause(0.05)
                

                
        i += 1
        if msvcrt.kbhit() and msvcrt.getch()[0] == 27:
            aborted = True

    plt.show(block=False)

if __name__ == '__main__':
    main()