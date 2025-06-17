import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
import sys
import argparse
from data import create_dataset
from data.universal_dataset import AlignedDataset_all
from src.model_meanflow_v1 import (MFDiT,Trainer, MeanFlow,set_seed)
from src.UnetRes_Meanflow import UnetRes
def parsr_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", type=str, default='/gz-data/Dataset')
    parser.add_argument("--phase", type=str, default='test')
    parser.add_argument("--max_dataset_size", type=int, default=float("inf"))
    parser.add_argument('--load_size', type=int, default=258, help='scale images to this size') #568
    parser.add_argument('--crop_size', type=int, default=256, help='then crop to this size')
    parser.add_argument('--direction', type=str, default='AtoB', help='AtoB or BtoA')
    parser.add_argument('--preprocess', type=str, default='none', help='scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none]')
    parser.add_argument('--no_flip', type=bool, default=True, help='if specified, do not flip the images for data augmentation')
    parser.add_argument("--bsize", type=int, default=2)
    opt = parser.parse_args()
    return opt

sys.stdout.flush()
set_seed(10)

save_and_sample_every = 1000
if len(sys.argv) > 1:
    sampling_timesteps = int(sys.argv[1])
else:
    sampling_timesteps = 5
train_num_steps = 100000

condition = True

train_batch_size = 1
num_samples = 1
image_size = 256


opt = parsr_args()

results_folder = "./ckpt_universal/diffuir"

dataset = AlignedDataset_all(opt, image_size, augment_flip=False, equalizeHist=True, crop_patch=False, generation=False, task='blur')
num_unet = 1
objective = 'pred_res'
test_res_or_noise = "res"
sampling_timesteps = 10
sum_scale = 0.01
ddim_sampling_eta = 0.
delta_end = 1.8e-3

base_model = UnetRes(
    dim=64,
    dim_mults=(1, 2, 4, 8),
    num_unet=num_unet,
    condition=condition,
    objective=objective,
    test_res_or_noise = test_res_or_noise
)
# model = UnetRes(
#     dim=32,
#     dim_mults=(1, 1, 1, 1),
#     num_unet=num_unet,
#     condition=condition,
#     objective=objective,
#     test_res_or_noise = test_res_or_noise
# )
meanflow= MeanFlow(
    base_model,
    channels=3,
    image_size=image_size,
    flow_ratio=0.50,
    time_dist=['lognorm', -0.4, 1.0],
    cfg_ratio=0.10,
    cfg_scale=2.0,
    # experimental
    cfg_uncond='u')


trainer = Trainer(
    meanflow,
    dataset,
    opt,
    train_batch_size=train_batch_size,
    train_lr=8e-4,
    train_num_steps=train_num_steps,         # total training steps
    ema_decay=0.995,                # exponential moving average decay
    amp=False,                        # turn on mixed precision
    convert_image_to="RGB",
    results_folder = results_folder,
    condition=condition,
    save_and_sample_every=save_and_sample_every
)

# test
if not trainer.accelerator.is_local_main_process:
    pass
else:
    trainer.load(30)
    trainer.set_results_folder('./result_blur')
    trainer.test(last=True)
