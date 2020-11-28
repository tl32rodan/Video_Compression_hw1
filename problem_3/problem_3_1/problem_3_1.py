import numpy as np
import os
import torch
import torch.nn.functional as F
import time
from tqdm.autonotebook import tqdm
import json
import copy
import matplotlib.pyplot as plt


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


def find_best_fit_block(source_block, target_blocks, unit_size=(4,4), device=torch.device('cpu')):
    '''
    source_block  : torch.Tensor with dimention (unit_size)
    target_blocks : 2D array of torch.Tensor, each element has dimention (unit_size)
    '''
    
    height, width, _, _ = target_blocks.size()
    
    # Calculate MSE
    mse = torch.sum((source_block-target_blocks)**2, (2,3)).view(-1)
    
    _, min_mse_idx = torch.topk(mse, 1, largest=False)
    
    min_mse_idx = min_mse_idx.cpu().numpy()
    
    result_idx = ((min_mse_idx//width).item(), (min_mse_idx % width).item())
    
    return result_idx


def spatial_Wiener(source_frame, reference_range=4,
                 L_x=352, L_y=288,
                 unit_size=(4,4), 
                 device=torch.device('cpu'),
                 result_frame = None):
    '''
        source_frame  : 2D Tensor, size=(L_y, L_x)
        
        reference_range: Integer, number of reference neibors
        ---
        result_frame: Location to save result ; 
                        just for reducing works of moving data to GPU memory
    '''
    
    return_result = False
    if result_frame == None:
        result_frame = torch.zeros_like(source_frame).to(device)
        return_result = True
        
    filter_list = []

    # Extend source frame ; fill 128 around this frame so we don't need to deal with out-of-bound cases
    tmp_source_frame = torch.ones((L_y+2, L_x+2)).to(device)
    tmp_source_frame[1:-1, 1:-1] = source_frame
    
    # Calculate motion compesation for all blocks
    for h_base in range(L_y//unit_size[0]):
        for w_base in range(L_x//unit_size[1]):
            # Get current block from source_frame
            source_block = tmp_source_frame[h_base*unit_size[0]+1 : (h_base+1)*unit_size[0]+1,
                                            w_base*unit_size[1]+1 : (w_base+1)*unit_size[1]+1]

            predicted_blocks = []
            
            if reference_range >= 1:
                # Reference pixel : D, i.e. left-sided pixel
                predicted_blocks.append(tmp_source_frame[h_base*unit_size[0]+1 : (h_base+1)*unit_size[0]+1,
                                                         w_base*unit_size[1] : (w_base+1)*unit_size[1]])
            if reference_range >= 2:
                # Reference pixel : B, i.e. top-sided pixel
                predicted_blocks.append(tmp_source_frame[h_base*unit_size[0] : (h_base+1)*unit_size[0],
                                                         w_base*unit_size[1]+1 : (w_base+1)*unit_size[1]+1])
            if reference_range >= 4:
                # Reference pixel : A, i.e. top-left-sided pixel
                predicted_blocks.append(tmp_source_frame[h_base*unit_size[0] : (h_base+1)*unit_size[0],
                                                         w_base*unit_size[1] : (w_base+1)*unit_size[1]])
                
                # Reference pixel : C, i.e. top-right-sided pixel
                predicted_blocks.append(tmp_source_frame[h_base*unit_size[0]+2 : (h_base+1)*unit_size[0]+2,
                                                         w_base*unit_size[1]+2 : (w_base+1)*unit_size[1]+2])
            
            # Perform Wiener prediction on predicted_blocks
            X = [x.reshape(-1) for x in predicted_blocks]
            X = torch.stack(X).cpu().numpy().T
            Y = source_block.reshape(-1).cpu().numpy()
            
            # W_o = R^(-1) \cdot p
            ## R = sum(np.multiply(X[i] X[i].T))
            ## p = sum()
            R = np.sum([x*(x.reshape(-1,1)) for x in X], axis=0)
            p = np.sum([y*x for (x,y) in zip(X,Y)], axis=0)
            try:
                W_optimal = np.dot(np.linalg.inv(R), p)
            except:
                W_optimal = np.ones(reference_range)/reference_range
            
            prediction = [np.dot(W_optimal,x) for x in X]
            prediction = torch.Tensor(prediction).view(unit_size).to(device)

            # Paste prediction onto its own location
            # Temporarily store compensation results in result_frame
            result_frame[h_base*unit_size[0] : (h_base+1)*unit_size[0],
                         w_base*unit_size[1] : (w_base+1)*unit_size[1]] = prediction

            # Recording
            filter_list.append(W_optimal)
            
    # Finish motion compensation
    
    if return_result:
        return result_frame, filter_list
    else:
        return filter_list


def establish_Wiener_filter_dict(Y, reference_range=4,
                                       unit_size=(4,4), 
                                       L_x = 352, L_y = 288, L_t = 100,
                                       MAX_REFERENCE_RANGE = 4,
                                       dict_filename='tmp.json', 
                                       device = torch.device('cpu')):
    
    
    if os.path.isfile(dict_filename): 
        # Use exist dict
        print('Found ',dict_filename)
        print('Start to load exist dict of Wiener filter predicted frames...')
        with open(dict_filename, 'r') as fp: 
            result_dict = json.load(fp)
            fp.close()
        print('Finish loading') 
        return result_dict
    
    
    print('Start to establish new dict of Wiener filter predicted frames...')
    print('File name : ', dict_filename)
   
    input_seq = torch.Tensor(Y).type(torch.float32).to(device)

    result_dict = {}

    result_frames = torch.zeros_like(input_seq)
    
    for i in tqdm(range(MAX_REFERENCE_RANGE, L_t), desc='m_t'):
        
            # *Poor naming here
            source_frame  = input_seq[i]

            filter_list= spatial_Wiener(source_frame, 
                                         reference_range=reference_range,
                                         L_x=L_x, L_y=L_y,
                                         unit_size=unit_size, 
                                         device=device,
                                         result_frame=result_frames[i])

            result_dict[i] = {
                'Predicted_frames' : result_frames[i].cpu().numpy(),
                'Filter_list' : filter_list
            }
        
    # Save results
    with open(dict_filename,'w') as f:
        tmp_dict = copy.deepcopy(result_dict)

        for key in tmp_dict:
            tmp_dict[key]['Predicted_frames'] = tmp_dict[key]['Predicted_frames'].astype('int').tolist()
            tmp_dict[key]['Filter_list'] = np.array(tmp_dict[key]['Filter_list']).tolist()

        json.dump(tmp_dict, f)
        f.close()
    
    
    print('Finish estiblishment')
    return result_dict        


def cal_information_entropy(Y, result_dict):
    residual_frame_list = []
    filter_list = []
    
    for frame_idx in result_dict.keys():
        residual_frame_list.append(Y[int(frame_idx)] - np.array(result_dict[frame_idx]['Predicted_frames']))
        filter_list.append(np.array(result_dict[frame_idx]['Filter_list']))
        

    # Normalize & numerize Wiener filter coefficients
    w_max = np.max(np.array(filter_list))
    w_min = np.min(np.array(filter_list))
    filter_list = ((np.array(filter_list) - w_min)*255/(w_max-w_min)).astype(int)

    total_bits = []
    
    # Calculate entropy of those side information one-by-one
    for idx, l in enumerate([residual_frame_list, filter_list]):
        total_bits.append([])
        l = np.array(l)
        num_all_symbols = l.size
        
        symbol_prob_dict = {}

        # Calculate probability of symbol over entire video
        for symbol in np.unique(l):
            num_current_symbol = len(l[l==symbol])
            prob_of_symbol = num_current_symbol / num_all_symbols
            symbol_prob_dict[symbol] = prob_of_symbol
        
        # Calculate bitrate by frame
        for frame_info in l:
            bpp = 0
            for symbol in symbol_prob_dict.keys():
                bpp += (np.log2(1/symbol_prob_dict[symbol])*len(frame_info[frame_info==symbol]))
            total_bits[idx].append(bpp)
            
    total_bits_count = int(sum([sum(i) for i in total_bits]))
    print('Total bits = ',total_bits_count)
    return total_bits


def plot_all_info(Y, frame_info_dict, 
                       N = (4,2,1), n = (4,8,16), 
                       save_fig = True,
                       save_fig_name = 'tmp',
                       MAX_REFERENCE_RANGE = 4):
    
    var_ratio_dict = {}
    bitrate_dict = {}
    
    x = np.arange(MAX_REFERENCE_RANGE, len(Y))
    
    for _N in N:
        dest_dir = 'N_'+str(_N)
        
        fig = plt.figure(figsize=(16,12))
        var_ratio_plt = fig.add_subplot(2,1,1)
        bitrate_plt = fig.add_subplot(2,1,2)
        
        print('_N = ',_N)
        for _n in n:
            var_ratio_dict[(_N, _n)] = []
            bitrate_dict[(_N, _n)] = []
            
            # Calculate variance ratio
            for y, r in zip(Y[MAX_REFERENCE_RANGE:], frame_info_dict[(_N, _n)]['Residual_frames']):
                var_ratio_dict[(_N, _n)].append(r.var()/y.var())
            # Record bpp by frame
            for frame_bpp in zip(*frame_info_dict[(_N, _n)]['Bitrate_list']):
                bitrate_dict[(_N, _n)].append(sum(frame_bpp))
                
            # Plot the line w.r.t time
            var_ratio_plt.plot(x, var_ratio_dict[(_N, _n)], label='n = '+str(_n))
            bitrate_plt.plot(x, bitrate_dict[(_N, _n)], label='n = '+str(_n))
            
        # Figure settings
        var_ratio_plt.set_title('Variance ratio(N = '+str(_N)+')')
        var_ratio_plt.set_xlabel('Frame Index')
        var_ratio_plt.set_ylabel('Variance Ratio')
        var_ratio_plt.legend()
        
        bitrate_plt.set_title('Bitrate(N = '+str(_N)+')')
        bitrate_plt.set_xlabel('Frame Index')
        bitrate_plt.set_ylabel('Bitrate')
        bitrate_plt.legend()
        
        if save_fig:
            plt.savefig(os.path.join(dest_dir, save_fig_name+'_info_N_'+str(_N)+'.png'))
            
        plt.show()


def run_3_1(yuv_filename, dict_folder_name, device=torch.device('cpu'), save_fig_name='tmp'):
    Y, U, V = read_yuv_video(yuv_filename)

    input_seq = torch.Tensor([[Y]]).type(torch.float32).to(device)
    
    # Predict 
    N = (4,2,1)
    n = (4,8,16)
    MAX_REFERENCE_RANGE = N[0]
    
    # Collect residual frames for figure drawing
    frame_info_dict = {}
    
    for _N in N:
        print('N = ',str(_N),' :')
        dest_dir = 'N_'+str(_N)
        if not os.path.exists(dest_dir):
            os.mkdir(dest_dir)
            
        for _n in n:
            print('    n = ',_n,' :')
            
            residual_frames = []
            
            dict_filename = 'N_'+str(_N)+'_n_'+str(_n)+'.json'
            dict_filename = os.path.join(dict_folder_name, dict_filename)
            result_dict = establish_Wiener_filter_dict(
                                   Y,
                                   reference_range=_N,
                                   unit_size=(_n, _n),
                                   dict_filename=dict_filename, 
                                   device=device)            
            
            for frame_idx in result_dict.keys():
                    residual_frames.append(Y[int(frame_idx)] - np.array(result_dict[frame_idx]['Predicted_frames']))
             
            # Statistics
            total_bit_list = cal_information_entropy(Y, result_dict)
            
            frame_info_dict[(_N,_n)] = {'Residual_frames':residual_frames,
                                        'Bitrate_list' : total_bit_list}
            
            
        print('---------------------')
            
    # Draw figure 
            
    plot_all_info(Y, frame_info_dict, 
                       N=N, n=n, 
                       save_fig=True,
                       dest_dir=dest_dir,
                       save_fig_name=save_fig_name)

# +
import json

def save_yuv_from_motion_dict(yuv_filename,
                              dict_filename = './save_2_3/AKIYO/N_4_n_16.json',
                              output_video_name = './MOBILE_N_4_n_16.yuv' ,
                              save_residual=False):
    
    Y, U, V = read_yuv_video(yuv_filename)

    with open(dict_filename, 'r') as fp: 
        result_dict = json.load(fp)
        fp.close()
    
    with open(output_video_name ,'wb') as f:
    
        num_frames, height, width = Y.shape
        
        _U = (np.ones((width*height//4))*128).reshape(-1).astype(np.uint8)
        _V = (np.ones((width*height//4))*128).reshape(-1).astype(np.uint8)
        
        for frame_idx in result_dict.keys():
            if save_residual:
                motion_comp = Y[int(frame_idx)] - np.array(result_dict[frame_idx]['Predicted_frames'])
            else:
                motion_comp = np.array(result_dict[frame_idx]['Predicted_frames'])
            
            f.write(motion_comp.reshape(-1).astype(np.uint8).tobytes())
            f.write(_U.tobytes())
            f.write(_V.tobytes())
                

        print('Finish writing ',output_video_name)
        f.close()
# -

if __name__ == '__main__':
    # Predict by Wiener filter
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  
    
    yuv_filename = '../../MOBILE_352x288_10.yuv'
    dict_folder_name = './save_3_1/MOBILE/'
    
    run_3_1(yuv_filename, dict_folder_name, device=device, save_fig_name='MOBILE')
    
    
    yuv_filename = '../../AKIYO_352x288_10.yuv'
    dict_folder_name = './save_3_1/AKIYO/'
    
    run_3_1(yuv_filename, dict_folder_name, device=device, save_fig_name='AKIYO')

    yuv_filename = '../../MOBILE_352x288_10.yuv'
    dict_filename = './save_3_1/MOBILE/N_4_n_4.json'
    output_video_name = './MOBILE_spatial_Wiener_N_4_n_4.yuv' 
    save_yuv_from_motion_dict(yuv_filename, dict_filename, output_video_name, save_residual=False)


