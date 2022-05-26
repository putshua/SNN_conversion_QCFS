import matplotlib.pyplot as plt
import matplotlib.animation as animation

from dv import NetworkFrameInput
import numpy as np
import torch
import cv2
import os

DVS_LABELS = {
    0: 'hand clap',
    1: 'right hand wave',
    2: 'left hand wave',
    3: 'right arm clockwise',
    4: 'right arm counter clockwise',
    5: 'left arm clockwise',
    6: 'left arm counter clockwise',
    7: 'arm roll',
    8: 'air drums',
    9: 'air guitar',
    10: 'other gestures',
}

class LiveModule:

    def __init__(self, model, accumulator, device):
        self.model = model
        self.accumulator = accumulator
        self.device = device
        self.x = np.arange(len(DVS_LABELS)) - 0.5
        self.y = np.zeros(len(self.x))

    def start(self, interval = 500, address = '127.0.0.1', port = 9466):
        self.load_animation(interval)
        self.stream(address, port)

    def load_animation(self, interval):
        fig, ax = plt.subplots()
        _, _, bar_container = ax.hist(self.x, self.y)
        ax.set_ylim(top=50)  # tunable ? 
        ax.xaxis.set_ticks(np.arange(len(DVS_LABELS) + 1))
        ax.xaxis.set_ticklabels(np.arange(len(DVS_LABELS) + 1))
        ani = animation.FuncAnimation(fig, self.live_hist(bar_container), interval=interval, blit=True)
        plt.show()

    def live_hist(self, bar_container):
        def animate(frame_number):
            for count, rect in zip(self.y, bar_container.patches):
                rect.set_height(count)
            return bar_container.patches
        return animate

    def stream(self, address, port):
        with NetworkFrameInput(address=address, port=port) as frames:
            idx = 0
            spikes_buffer = 0
            spikes_mem = []
            self.model = self.model.cuda(self.device)

            for frame in frames:
                # create image
                f = frame.image - 127.0
                channel1 = np.clip(f, 0, 128) 
                channel2 = np.clip(f, -127, 0)
                image = np.concatenate([channel1, channel2], axis=2)

                # process image
                image = torch.from_numpy(image)
                image = image.unsqueeze(0)
                image = torch.permute(image, (0, 3, 1, 2))
                image = image[:, :, :128, :128]
                image = image.cuda()

                # call model
                with torch.no_grad():
                    output = self.model(image)

                # store result
                spikes_buffer += output
                spikes_mem.append(output)
                if idx > self.accumulator:
                    spikes_buffer -= spikes_mem.pop(0)
                
                # get prediction
                pred = spikes_buffer[:, :10].max(1)[1]
                self.y = spikes_buffer[:, :10].tolist()

                # display results
                if idx % 100 == 0: 
                    os.system('clear')
                    print(spikes_buffer[:, :10])
                    print(DVS_LABELS[pred.item()])
            
                idx += 1