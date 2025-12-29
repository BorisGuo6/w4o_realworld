import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
from PIL import Image
import torch.nn.functional as F
from utils.utils_correspondence import resize
from model_utils.extractor_sd import load_model, process_features_and_mask
from model_utils.extractor_dino import ViTExtractor
from model_utils.projection_network import AggregationNetwork
import numpy as np
from utils.utils_visualization_demo import Demo
import pickle
import matplotlib.pyplot as plt
import argparse

num_patches = 60
sd_model = sd_aug = extractor_vit = None
aggre_net = AggregationNetwork(feature_dims=[640,1280,1280,768], projection_dim=768, device='cuda')
aggre_net.load_pretrained_weights(torch.load('results_spair/best_856.PTH'))

def random_uv_from_segmentation(segmentation: np.ndarray):
    # 获取 segmentation 中大于 0 的像素位置
    indices = np.argwhere(segmentation > 0.5)
    indices = [tuple(indice) for indice in indices]
    
    permuted_indices = np.random.permutation(indices)
    
    # 返回随机选择的 uv 坐标
    return permuted_indices
        
def get_processed_features(sd_model, sd_aug, aggre_net, extractor_vit, num_patches, img=None, img_path=None):
    
    if img_path is not None:
        feature_base = img_path.replace('JPEGImages', 'features').replace('.jpg', '')
        sd_path = f"{feature_base}_sd.pt"
        dino_path = f"{feature_base}_dino.pt"

    # extract stable diffusion features
    if img_path is not None and os.path.exists(sd_path):
        features_sd = torch.load(sd_path)
        for k in features_sd:
            features_sd[k] = features_sd[k].to('cuda')
    else:
        if img is None: img = Image.open(img_path).convert('RGB')
        img_sd_input = resize(img, target_res=num_patches*16, resize=True, to_pil=True)
        features_sd = process_features_and_mask(sd_model, sd_aug, img_sd_input, mask=False, raw=True)
        del features_sd['s2']

    # extract dinov2 features
    if img_path is not None and os.path.exists(dino_path):
        features_dino = torch.load(dino_path)
    else:
        if img is None: img = Image.open(img_path).convert('RGB')
        img_dino_input = resize(img, target_res=num_patches*14, resize=True, to_pil=True)
        img_batch = extractor_vit.preprocess_pil(img_dino_input)
        features_dino = extractor_vit.extract_descriptors(img_batch.cuda(), layer=11, facet='token').permute(0, 1, 3, 2).reshape(1, -1, num_patches, num_patches)

    desc_gathered = torch.cat([
            features_sd['s3'],
            F.interpolate(features_sd['s4'], size=(num_patches, num_patches), mode='bilinear', align_corners=False),
            F.interpolate(features_sd['s5'], size=(num_patches, num_patches), mode='bilinear', align_corners=False),
            features_dino
        ], dim=1)
    
    desc = aggre_net(desc_gathered) # 1, 768, 60, 60
    # normalize the descriptors
    norms_desc = torch.linalg.norm(desc, dim=1, keepdim=True)
    desc = desc / (norms_desc + 1e-8)
    return desc

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Depth Anything V2 Metric Depth Estimation')
    
    parser.add_argument('--img-path', type=str, default='./pcd_data') # path to the source image
    parser.add_argument('--threshold', type=float, default=0.605)
    # parser.add_argument('--save-path', type=str, default='./matchings')
    
    args = parser.parse_args()

    img_size = (200, 200) # (H, W)
    sd_model, sd_aug = load_model(diffusion_ver='v1-5', image_size=num_patches*16, num_timesteps=50, block_indices=[2,5,8,11])
    extractor_vit = ViTExtractor('dinov2_vitb14', stride=14, device='cuda')
    # img1_path = os.path.join(args.img_path, '1.png') # path to the source image
    img1_path = os.path.join(args.img_path, 'rgb_init.png') # path to the source image
    img1 = Image.open(img1_path).convert('RGB').resize(img_size)# resize(Image.open(img1_path).convert('RGB'), target_res=img_size, resize=True, to_pil=True) #[W, H, 3]]

    # img2_path = os.path.join(args.img_path, '2.png') # path to the source image
    img2_path = os.path.join(args.img_path, 'rgb_goal.png') # path to the source image
    img2 = Image.open(img2_path).convert('RGB').resize(img_size)# resize(Image.open(img2_path).convert('RGB'), target_res=img_size, resize=True, to_pil=True)

    feat1 = get_processed_features(sd_model, sd_aug, aggre_net, extractor_vit, num_patches, img=img1)
    feat2 = get_processed_features(sd_model, sd_aug, aggre_net, extractor_vit, num_patches, img=img2)
    
    # seg1_path = os.path.join(args.img_path, '1_mask.png') # path to the source image
    seg1_path = os.path.join(args.img_path, 'mask_init.png') # path to the source image
    seg1 = np.array(Image.open(seg1_path).convert('1').resize(img_size)) # [H, W]
    # seg2_path = os.path.join(args.img_path, '2_mask.png') # path to the source image
    seg2_path = os.path.join(args.img_path, 'mask_goal.png') # path to the source image
    seg2 = np.array(Image.open(seg2_path).convert('1').resize(img_size))

    demo = Demo([img1,img2], torch.cat([feat1, feat2], dim=0), img_size)
    src_uvs = random_uv_from_segmentation(np.array(seg1))
    point_wise_matching = {}
    # pair_similarity = {}
    num_of_matchings = 1000
    
    temp = 0
    current_matchings = 0
    print(f"num_possible_matchings: {len(src_uvs)}")
    # while temp<=500:
    #     if temp >= len(src_uvs):
    #         break
    for temp in range(min(len(src_uvs),num_of_matchings)):
        (x, y) = src_uvs[temp]
        tgt_uv, similarity = demo.find_matching((x, y), seg2, threshold=0.7)
        # import pdb
        # pdb.set_trace()
        # pair_similarity[(tuple(src_uvs[temp]), tgt_uv)] = similarity
        (tgt_x, tgt_y) = tgt_uv if tgt_uv is not None else (-1, -1)
        if similarity > args.threshold and np.sqrt((tgt_x - x)**2 + (tgt_y - y)**2) >= 0.1*img_size[0]:
            point_wise_matching[(x, y)] = tgt_uv
        # if tgt_uv is not None:
        #     # temp += 1
        #     if temp%100 == 0:
        #         print(f"src_uv: {x, y}, tgt_uv: {tgt_uv}")
        #     if seg2[tgt_uv[0], tgt_uv[1]] > 0.5:
        #         point_wise_matching[(x, y)] = tgt_uv
        #         current_matchings += 1
        #     if current_matchings > num_of_matchings:
        #         break
    # top_keys = list(sorted(point_wise_similarity, key=point_wise_similarity.get, reverse=True))
    # for pair in top_keys[:min(len(top_keys), 200)]:
    #     point_wise_matching[pair[0]] = pair[1]
    #     print(point_wise_similarity[pair])
    print(f"num_point_wise_matching: {len(point_wise_matching)}")
    srcs = list(point_wise_matching.keys())
    tgts = list(point_wise_matching.values())

    save_path = os.path.join(args.img_path, 'matching_results')
    os.makedirs(save_path, exist_ok=True)
    for i in range(min(20, len(srcs))):
        fig, axes = plt.subplots(1, 2, figsize=(3*2, 3))

        plt.tight_layout()
        axes[0].imshow(img1)
        axes[0].axis('off')
        axes[0].scatter(srcs[i][1], srcs[i][0], s=10, c='r')
        axes[1].imshow(img2)
        axes[1].axis('off')
        axes[1].scatter(tgts[i][1], tgts[i][0], s=10, c='b')

        plt.savefig(os.path.join(save_path, f'matching_{i}.png'))
        plt.close()

    pickle.dump(point_wise_matching, open(os.path.join(args.img_path, 'point_wise_matching.pkl'), 'wb'))