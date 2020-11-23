from problem_1_1 import autocorrelation
from problem_1_2 import run_1_2
from problem_1_3 import run_1_3
import json
import copy

# ## Run (1)-2

# +
yuv_filename = '../AKIYO_352x288_10.yuv'
output_video_name = './AKIYO_AC.yuv'

run_1_2(yuv_filename, output_video_name)
# -

# ## Run (1)-3

yuv_filename = '../MOBILE_352x288_10.yuv'
output_video_name = './Mobile_AC_motion.yuv'
motion_result_dict = run_1_3(yuv_filename, output_video_name)


print(motion_result_dict[0]['Motion_vector_lists'][0])

# Save results
with open('./motion_ckpt/motion_result_dict_Mobile.json','w') as f:
    tmp_dict = copy.deepcopy(motion_result_dict)

    for key in tmp_dict:
         tmp_dict[key]['Predict_frames'] = tmp_dict[key]['Predict_frames'].cpu().numpy().astype('int').tolist()
#         tmp_dict[key]['Motion_vector_lists'] = [torch.stack(l) for l in tmp_dict[key]['Motion_vector_lists']]
#         tmp_dict[key]['Motion_vector_lists'] = torch.stack(tmp_dict[key]['Motion_vector_lists'])
#         tmp_dict[key]['Motion_vector_lists'] = tmp_dict[key]['Motion_vector_lists'].cpu().numpy().tolist()

    json.dump(tmp_dict, f)
    f.close()


