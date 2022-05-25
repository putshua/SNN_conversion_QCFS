from dv import NetworkFrameInput
import numpy as np
import torch

DVS_LABELS = {
    1: 'hand clap',
    2: 'right hand wave',
    3: 'left hand wave',
    4: 'right arm clockwise',
    5: 'right arm counter clockwise',
    6: 'left arm clockwise',
    7: 'left arm counter clockwise',
    8: 'arm roll',
    9: 'air drums',
    10: 'air guitar',
    11: 'other gestures',
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
                pred = spikes_buffer.max(1)[1]

                # print pred
                # ToDo: Showcase confidencence distribution
                print(DVS_LABELS[pred.item()])

                # print live images 
                # ToDo: elaborate something fancier
                # ToDo: add a way to stop the simulation

            
                idx += 1