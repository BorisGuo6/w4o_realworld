"""borrowed from https://github.com/Tsingularity/dift/blob/main/src/utils/visualization.py"""

import gc
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class Demo:

    def __init__(self, imgs, ft, img_size, dist='argmax'):
        self.ft = ft # NCHW
        self.imgs = imgs
        self.num_imgs = len(imgs)
        self.img_size = img_size
        self.dist = dist
        self.cos = nn.CosineSimilarity(dim=1)
        
    def find_matching(self, src_uv: tuple, seg: np.ndarray, threshold: float = 0.7):
        """
        Find the matching point in the target image based on the source point.
        Args:
            src_uv: (x, y) coordinates of the source point in the source image.
            seg: (H, W) segmentation map of the target image, 1 for targeted areas, otherwise 0.
            threshold: float, threshold for cosine similarity.
        Returns:
            trg_uv: (x, y) coordinates of the target point in the target image.
        """
        with torch.no_grad():
            H, W = seg.shape # 480, 640
            num_channel = self.ft.size(1)
            src_ft = self.ft[0].unsqueeze(0)
            src_ft = nn.Upsample(size=(H, W), mode='bilinear')(src_ft)
            src_vec = src_ft[0, :, src_uv[0], src_uv[1]].view(1, num_channel, 1, 1)  # 1, C, 1, 1

            del src_ft
            gc.collect()
            torch.cuda.empty_cache()
            # import pdb
            # pdb.set_trace()

            trg_ft = nn.Upsample(size=(H, W), mode='bilinear')(self.ft[1:]) # 1, C, H, W
            cos_map = self.cos(src_vec, trg_ft).detach().cpu().numpy()    # 1, H, W
            
            # Mask out non-related pixels
            cos_map = cos_map - 100*(1.-seg[None,...])
            del trg_ft
            gc.collect()
            torch.cuda.empty_cache()
            for i in range(1, self.num_imgs):
                max_xy = np.unravel_index(cos_map[i-1].argmax(), cos_map[i-1].shape)
            
        return (max_xy[0], max_xy[1]), cos_map[i-1].max()  # (x, y) coordinates of the target point in the target image

    def plot_img_pairs(self, fig_size=3, alpha=0.45, scatter_size=70, seg=None):

        fig, axes = plt.subplots(1, self.num_imgs, figsize=(fig_size*self.num_imgs, fig_size))

        plt.tight_layout()

        for i in range(self.num_imgs):
            axes[i].imshow(self.imgs[i])
            axes[i].axis('off')
            if i == 0:
                axes[i].set_title('source image')
            else:
                axes[i].set_title('target image')

        num_channel = self.ft.size(1)
        cos = nn.CosineSimilarity(dim=1)
        
        W, H = self.imgs[0].size[0], self.imgs[0].size[1] # 640, 480
        print(H, W)

        def onclick(event):
            if event.inaxes == axes[0]:
                with torch.no_grad():
                    
                    x, y = int(np.round(event.xdata)), int(np.round(event.ydata))

                    src_ft = self.ft[0].unsqueeze(0)
                    src_ft = nn.Upsample(size=(H, W), mode='bilinear')(src_ft)
                    src_vec = src_ft[0, :, y, x].view(1, num_channel, 1, 1)  # 1, C, 1, 1

                    del src_ft
                    gc.collect()
                    torch.cuda.empty_cache()

                    trg_ft = nn.Upsample(size=(H, W), mode='bilinear')(self.ft[1:]) # 1, C, H, W
                    if self.dist == 'argmax':
                        cos_map = cos(src_vec, trg_ft).cpu().numpy()    # 1, H, W
                        
                    del trg_ft
                    gc.collect()
                    torch.cuda.empty_cache()

                    axes[0].clear()
                    axes[0].imshow(self.imgs[0])
                    axes[0].axis('off')
                    axes[0].scatter(x, y, c='r', s=scatter_size)
                    axes[0].set_title('source image')
                    original_heat_map = cos_map

                    for i in range(1, self.num_imgs):
                        if seg is not None:
                            cos_map[i-1] = cos_map[i-1] - 100*(1.-seg[None,...])
                        if self.dist == 'argmax':
                            max_yx = np.unravel_index(cos_map[i-1].argmax(), cos_map[i-1].shape)

                        axes[i].clear()

                        heatmap = original_heat_map[i-1]
                        print(heatmap.shape)
                        heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))  # Normalize to [0, 1]
                        axes[i].imshow(self.imgs[i])
                        axes[i].imshow(255 * heatmap, alpha=alpha, cmap='viridis')
                        axes[i].axis('off')
                        axes[i].scatter(max_yx[1], max_yx[0], c='r', s=scatter_size)
                        axes[i].set_title('target image')

                    del cos_map
                    del heatmap
                    gc.collect()

        fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()
