import cv2
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("CUDA is available:", torch.cuda.is_available())

from segment_anything import (SamAutomaticMaskGenerator, sam_model_registry)

import gc
gc.collect()
torch.cuda.empty_cache()

image = cv2.imread('src/images/dog.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

print('Original Dimensions : ', image.shape)
 
scale_percent = 25 # percent of original size
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)

image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
print('Resized Dimensions : ', image.shape)

# plt.figure(figsize=(5, 5))
# plt.imshow(image)
# plt.axis('off')
# plt.show()

def show_anns(anns):

    if len(anns) == 0:
        return

    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []

    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.35)))

sam = sam_model_registry["vit_h"](checkpoint="src/sam_vit_h_4b8939.pth")

# sam.to(device="cuda")

mask_generator = SamAutomaticMaskGenerator(sam)

masks = mask_generator.generate(image)

plt.figure(figsize=(5, 5))
plt.imshow(image)
show_anns(masks)
plt.axis('off')
plt.show()
