from dv import NetworkFrameInput
import numpy as np
import cv2

with NetworkFrameInput(address='127.0.0.1', port=9466) as i:
    idx = 0
    for frame in i:
        f = frame.image - 127.0
        chan1 = np.clip(f, 0, 128)
        chan2 = -np.clip(f, -127, 0)
        if idx == 10:
            breakpoint()
        im = np.concatenate([chan1, chan2], axis=2)
        cv2.imshow('out', chan1)
        cv2.imshow('out2', chan2)
        cv2.waitKey(100)
        idx+=1