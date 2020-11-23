# ## 2.

# +
import numpy as np
import os
import torch
import torch.nn.functional as F
import time
from tqdm.autonotebook import tqdm

from problem_1_1 import autocorrelation


# -

def read_yuv_video(yuv_filename):
    with open(yuv_filename ,'rb') as f:
        width, height, n_frames = 352, 288, 100

        Y = []
        U = []
        V = []

        for i in range(n_frames):
            yuv_frame = np.frombuffer(f.read(width*height*3//2), dtype=np.uint8)
            Y.append(np.array(yuv_frame[:width*height]).reshape(height, width))
            U.append(np.array(yuv_frame[width*height:-width*height//4]))
            V.append(np.array(yuv_frame[-width*height//4:]))

        Y = np.array(Y)
        U = np.array(U)
        V = np.array(V)
        f.close()
    return Y,U,V


def run_1_2(yuv_filename, output_video_name):
    # Read video
    Y, U, V = read_yuv_video(yuv_filename)
    
    # -------------------------------------------------------------------------
    # Normalize each frame with its own mean before calculating autocorrelation
    # -------------------------------------------------------------------------
    _Y = []
    for frame in Y:
        _Y.append(frame-frame.mean())

    Y = np.array(_Y)
    
    # -----------------------------------------------------
    # Calculate autocorrelation ; result is stored in R_xxx
    # -----------------------------------------------------
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')     

    R_xxx = np.zeros([21,144,176])

    input_seq = torch.Tensor([[Y]]).type(torch.float32).to(device)


    print('Start calculating autocorrelation...')
    for m_t in tqdm(range(10+1), desc='m_t'):
        m_t_fill = [m_t, 20-m_t]

        #for m_y in tqdm(range(72+1), desc='m_y', leave=False):
        for m_y in range(72+1):
            m_y_fill = [m_y]
            if m_y < 72 and m_y !=0:
                m_y_fill.append(144-m_y)
    
            #for m_x in tqdm(range(88+1), desc='m_x',leave=False):
            for m_x in range(88+1):
                m_x_fill = [m_x]
                if m_x < 88 and m_x !=0:
                    m_x_fill.append(176-m_x)

                # Calculation
                result = autocorrelation(input_seq, m_x-88, m_y-72, m_t-10, device=device)
                
                for x in m_x_fill:
                    for y in m_y_fill:
                        for t in m_t_fill:
                            R_xxx[t,y,x] = result
            
    
    print('Finish calculating autocorrelation.')
    print('R_xxx[0,0,0]= ', R_xxx[10,72,88])
                                       
    # ----------------------------------------------------------------
    # Normalize with R_xxx[0,0,0] 
    # Since we have shifted R_xxx, R_xxx[10,72,88] is actually R_xxx[0,0,0]
    # ----------------------------------------------------------------
    _Y = R_xxx

    norm = R_xxx[10,72,88]
    _Y = (_Y/norm)*127.5+127.5
    _Y = np.cast['uint8'](_Y)
    
    # ------------------------
    # Store R_xxx as an video
    # ------------------------
    with open(output_video_name ,'wb') as f:

        width, height, n_frames = 176, 144, 21

        _U = (np.ones((width*height//4))*128).reshape(-1).astype(np.uint8)
        _V = (np.ones((width*height//4))*128).reshape(-1).astype(np.uint8)

        #_Y = R_xxx
        for frame in _Y:
            f.write(frame.reshape(-1).astype(np.uint8).tobytes())
            f.write(_U.tobytes())
            f.write(_V.tobytes())
        f.close()
    print('Finish writing ',output_video_name)

if __name__=='__main__':
    yuv_filename = '../MOBILE_352x288_10.yuv'
    output_video_name = './Mobile_AC.yuv'
    
    run_1_2(yuv_filename, output_video_name)


    yuv_filename = '../AKIYO_352x288_10.yuv'
    output_video_name = './AKIYO_AC.yuv'
    
    run_1_2(yuv_filename, output_video_name)


