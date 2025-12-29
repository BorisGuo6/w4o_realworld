import cv2
import torch
import sys
sys.path.insert(0, str('Depth-Anything-V2/metric_depth')) # The `generate_metric_depth.py` should run under `Depth-Anything-V2/metric_depth`.

from depth_anything_v2.dpt import DepthAnythingV2
import matplotlib.pyplot as plt
import argparse
import os
import glob

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2 Metric Depth Estimation')
    
    parser.add_argument('--img-path', type=str)
    
    parser.add_argument('--encoder', type=str, default='vitb', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--load-from', type=str, default='/home/ubuntu/feature_based_icp/Depth-Anything-V2/metric_depth/checkpoints/depth_anything_v2_metric_hypersim_vitl.pth')
    parser.add_argument('--max-depth', type=float, default=20)
    
    args = parser.parse_args()

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
    }

    encoder = 'vitl' # or 'vits', 'vitb'
    dataset = 'hypersim' # 'hypersim' for indoor model, 'vkitti' for outdoor model
    max_depth = 20 # 20 for indoor model, 80 for outdoor model

    model = DepthAnythingV2(**{**model_configs[encoder]})
    model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_metric_{dataset}_{encoder}.pth', map_location='cpu'))
    model.cuda().eval()

    # img_names = glob.glob(os.path.join(args.img_path, '*.png'))
    # mask_names = glob.glob(os.path.join(args.img_path, '*_*.png'))
    # imgs = []
    # for img_name in img_names:
    #     if img_name not in mask_names:
    #         imgs.append(img_name)
    img_names = glob.glob(os.path.join(args.img_path, 'rgb_*.png'))
    mask_names = glob.glob(os.path.join(args.img_path, 'mask_*.png'))

    for img_name in img_names:
        # name1, name2 = os.path.splitext(img_name)
        raw_img = cv2.imread(img_name)
        depth = model.infer_image(raw_img) # HxW depth map in meters in numpy
        plt.imshow(depth, cmap='gray')
        plt.axis('off')
        # plt.savefig(name1+'_dpt.png', bbox_inches='tight', pad_inches=0)
        plt.savefig(img_name.replace('rgb_', 'dpt_'), bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f'Progressed {img_name}')