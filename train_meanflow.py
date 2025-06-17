import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
import sys
import argparse
from data import create_dataset
from data.universal_dataset import AlignedDataset_all
from src.model_meanflow_v1 import Trainer,MeanFlow,set_seed
from src.UnetRes_Meanflow import UnetRes
import torch
import torch.nn as nn
def parsr_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", type=str, default='/gz-data/Dataset')
    parser.add_argument("--phase", type=str, default='train')
    parser.add_argument("--max_dataset_size", type=int, default=float("inf"))
    parser.add_argument('--load_size', type=int, default=258, help='scale images to this size') #572,268
    parser.add_argument('--crop_size', type=int, default=256, help='then crop to this size')
    parser.add_argument('--direction', type=str, default='AtoB', help='AtoB or BtoA')
    parser.add_argument('--preprocess', type=str, default='crop', help='scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none]')
    parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
    parser.add_argument("--bsize", type=int, default=2)
    opt = parser.parse_args()
    return opt

sys.stdout.flush()
set_seed(10)

save_and_sample_every = 1000
if len(sys.argv) > 1:
    sampling_timesteps = int(sys.argv[1])
else:
    sampling_timesteps = 10

train_batch_size = 10
num_samples = 1
sum_scale = 0.01
image_size = 256
condition = True
opt = parsr_args()

results_folder = "./ckpt_universal/diffuir"

if 'universal' in results_folder:
    #dataset_fog = AlignedDataset_all(opt, image_size, augment_flip=True, equalizeHist=True, crop_patch=True, generation=False, task='fog')
    dataset_light = AlignedDataset_all(opt, image_size, augment_flip=True, equalizeHist=True, crop_patch=True, generation=False, task='light_only')
    dataset_rain = AlignedDataset_all(opt, image_size, augment_flip=True, equalizeHist=True, crop_patch=True, generation=False, task='rain')
    dataset_snow = AlignedDataset_all(opt, image_size, augment_flip=True, equalizeHist=True, crop_patch=True, generation=False, task='snow')
    dataset_blur = AlignedDataset_all(opt, image_size, augment_flip=True, equalizeHist=True, crop_patch=True, generation=False, task='blur')
    dataset = [dataset_light, dataset_rain, dataset_snow, dataset_blur]
    
    num_unet = 1
    objective = 'pred_res'
    test_res_or_noise = "res"
    train_num_steps = 30000
    train_batch_size = 10
    sum_scale = 0.01
    delta_end = 1.8e-3



base_model = UnetRes(
    dim=64,
    dim_mults=(1, 2, 4, 8),
    num_unet=num_unet,
    condition=condition,
    objective=objective,
    test_res_or_noise = test_res_or_noise
)
meanflow= MeanFlow(
    base_model,
    channels=3,
    image_size=image_size,
    flow_ratio=0.50,
    time_dist=['lognorm', -0.4, 1.0],
    cfg_ratio=0,
    cfg_scale=2.0,
    # experimental
    cfg_uncond='u',jvp_api="funtorch")


trainer = Trainer(
    meanflow,
    dataset,
    opt,
    train_batch_size=train_batch_size,
    train_lr=8e-5,
    train_num_steps=train_num_steps,         # total training steps
    ema_decay=0.995,                # exponential moving average decay
    amp=False,                        # turn on mixed precision
    fp16=False,
    convert_image_to="RGB",
    results_folder = results_folder,
    condition=condition,
    save_and_sample_every=save_and_sample_every
)
# train
# trainer.load(30)
trainer.train()