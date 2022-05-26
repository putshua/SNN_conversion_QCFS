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

    def start(self, address = '127.0.0.1', port = 9466):
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
                
                image = torch.from_numpy(image)
                image = image.unsqueeze(0)
                image = torch.permute(image, (0, 3, 1, 2))
                image = image[:, :, :128, :128]

                # cv2.imshow('image test', image[0, 0, :, :].numpy())
                # cv2.waitKey(100)
                # print(image)

                # print(image.numpy())
                image = image.cuda()


                # call model
                with torch.no_grad():
                    output = self.model(image)

                # store result
                spikes_buffer += output
                spikes_mem.append(output)
                if idx > self.accumulator:
                    spikes_buffer -= spikes_mem.pop(0)
                
                # translate pred

                # ToDo: retrieve confidence distribution
                # Question: what's happening here ?

                pred = spikes_buffer[:, :10].max(1)[1]

                # print pred
                # ToDo: Showcase confidencence distribution
                if idx%100 == 0: 
                    os.system('clear')

                    print(spikes_buffer[:, :10])
                    print(DVS_LABELS[pred.item()])


                # print live images 
                # ToDo: elaborate something fancier
                # ToDo: add a way to stop the simulation

            
                idx += 1