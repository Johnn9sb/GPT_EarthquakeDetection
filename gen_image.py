import os 
import sys
import torch
import seisbench.data as sbd
import seisbench.generate as sbg
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import time

# =========================================================================================================
# DataLoad
start_time = time.time()
# CWBSN load
cwb = sbd.WaveformDataset('/mnt/nas2/weiwei/seisbench_cache/datasets/cwbsn',sampling_rate=100)
c_mask = cwb.metadata["trace_completeness"] == 4
cwb.filter(c_mask)
ctrain, _, _ = cwb.train_dev_test()
# NOISE load
# Combine  
noise = sbd.WaveformDataset("/mnt/nas2/weiwei/seisbench_cache/datasets/cwbsn_noise",sampling_rate=100)
ntrain, _, _ = noise.train_dev_test()

end_time = time.time()
elapsed_time = end_time - start_time
print("=====================================================")
print(f"Load data time: {elapsed_time} sec")
print("=====================================================")
print("Data loading complete!!!")

phase_dict = {
        "trace_p_arrival_sample": "P",
        "trace_pP_arrival_sample": "P",
        "trace_P_arrival_sample": "P",
        "trace_P1_arrival_sample": "P",
        "trace_Pg_arrival_sample": "P",
        "trace_Pn_arrival_sample": "P",
        "trace_PmP_arrival_sample": "P",
        "trace_pwP_arrival_sample": "P",
        "trace_pwPm_arrival_sample": "P",
        "trace_s_arrival_sample": "S",
        "trace_S_arrival_sample": "S",
        "trace_S1_arrival_sample": "S",
        "trace_Sg_arrival_sample": "S",
        "trace_SmS_arrival_sample": "S",
        "trace_Sn_arrival_sample": "S",
    }

ptime = 1000
augmentations = [
    sbg.WindowAroundSample(list(phase_dict.keys()), samples_before=3000, windowlen=9000, selection="first", strategy="pad"),
    # sbg.RandomWindow(windowlen=6000, strategy="pad",low=0,high=6000),
    sbg.FixedWindow(p0=2000,windowlen=6000,strategy="pad"),
    sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type="peak"),
    sbg.Filter(N=5, Wn=[1,10],btype='bandpass'),
    sbg.ChangeDtype(np.float32),
    sbg.ProbabilisticLabeller(label_columns=phase_dict, sigma=20, dim=0)
]
cwb_gene = sbg.GenericGenerator(ctrain)
cwb_gene.add_augmentations(augmentations)
cwb_loader = DataLoader(cwb_gene,batch_size=1, shuffle=True, num_workers=4,pin_memory=True)
noise_gene = sbg.GenericGenerator(ntrain)
noise_gene.add_augmentations(augmentations)
noise_loader = DataLoader(noise_gene,batch_size=1, shuffle=True, num_workers=4,pin_memory=True)

print("Dataloader Complete!!!")

cwb_gen = tqdm(cwb_loader, total = len(cwb_loader))
noise_gen = tqdm(noise_loader, total = len(noise_loader))

# image = 0
# for batch in cwb_gen:
#     x = batch['X']
#     wav1 = x[0,0]
#     wav2 = x[0,1]
#     wav3 = x[0,2]

#     wav1 = wav1.detach().numpy()
#     wav2 = wav2.detach().numpy()
#     wav3 = wav3.detach().numpy()

#     wav1 = np.square(wav1)
#     wav2 = np.square(wav2)
#     wav3 = np.square(wav3)
    
#     wav1 = wav1 + wav2 + wav3
#     wav1 = np.sqrt(wav1)

#     plt.figure(figsize=(15,5))
    
#     plt.subplot(111)  
#     plt.plot(wav1)
#     # plt.subplot(312)  
#     # plt.plot(wav2)
#     # plt.subplot(313)  
#     # plt.plot(wav3)
#     plt.tight_layout()
#     fignum1 = str(image)
#     savepath = '/mnt/nas5/johnn9/gptpicker/image/eq_syn/wav_' + fignum1 + '.jpg'
#     plt.savefig(savepath)
#     plt.close('all') 
#     image = image + 1
#     if image == 300:
#         break

image = 0
for batch in noise_gen:
    x = batch['X']
    wav1 = x[0,0]
    wav2 = x[0,1]
    wav3 = x[0,2]

    wav1 = wav1.detach().numpy()
    wav2 = wav2.detach().numpy()
    wav3 = wav3.detach().numpy()

    wav1 = np.square(wav1)
    wav2 = np.square(wav2)
    wav3 = np.square(wav3)
    
    wav1 = wav1 + wav2 + wav3
    wav1 = np.sqrt(wav1)

    plt.figure(figsize=(15,5))
    
    plt.subplot(111)  
    plt.plot(wav1)
    # plt.subplot(312)  
    # plt.plot(wav2)
    # plt.subplot(313)  
    # plt.plot(wav3)
    plt.tight_layout()
    fignum1 = str(image)
    savepath = '/mnt/nas5/johnn9/gptpicker/image/noise_syn/wav_' + fignum1 + '.jpg'
    plt.savefig(savepath)
    plt.close('all') 
    image = image + 1
    if image == 600:
        break
    