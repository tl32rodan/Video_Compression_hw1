# # Autocorrelation
# ## 1.

import numpy as np
import os
import torch
import torch.nn.functional as F
import time


def autocorrelation(input_seq, m_x, m_y, m_t, L_x=352, L_y=288, L_t=100, device=torch.device('cpu')):
    '''
        input_seq : numpy.ndarray(3d) or torch.Tensor(5d)
        m_x, m_y, m_t : Scalar
        L_x, L_y, L_t : Scalar
    ''' 
    
    if abs(m_x) >= L_x or abs(m_y) >= L_y or abs(m_t) >= L_t:
        return 0
    
    if type(input_seq) == np.ndarray and len(input_seq.shape) == 3:
        input_seq = torch.Tensor([[input_seq]]).type(torch.float32).to(device)
    elif type(input_seq) == torch.Tensor and len(input_seq.shape) == 5:
        pass
    else:
        raise ValueError
    
    inputs = input_seq[:, :, :L_t-abs(m_t),:L_y-abs(m_y),:L_x-abs(m_x)]
    filters = input_seq[:, :, abs(m_t):,abs(m_y):,abs(m_x):]
    #print(inputs.shape)
    
    #print(filters.shape)
    #start_t = time.time()
    
    #C_vvv = F.conv3d(inputs, filters).view(-1)[0].cpu().numpy()    
    C_vvv = (inputs*filters).sum().cpu().numpy()
    
    R_xxx = C_vvv / ((L_x-abs(m_x))*(L_y-abs(m_y))*(L_t-abs(m_t)))
    
    #end_t = time.time()
    #print('Autocorrelation value = ',R_xxx,' ; time = ',end_t-start_t)
    return R_xxx
