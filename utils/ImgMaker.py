import torch
import yaml
import numpy as np
import random
import cv2
from PIL import Image
import numpy as np
import json
from tqdm import tqdm
import os
import torchvision.transforms as transforms

import degradation as de
from RealESRNet_Dmodel import RealESRNetModel

def complex_degrade(img):
    with open('./Complex_cfg.yml', mode='r') as f:
        opt = yaml.load(f, Loader=yaml.FullLoader)

    model = RealESRNetModel(opt)
    
    kernel_list = opt["kernel_list"]
    kernel_prob = opt["kernel_prob"]
    kernel_size = 5 
    blur_sigma = opt['blur_sigma']
    betag_range = opt['betag_range']
    betap_range = opt['betap_range']

    trans = transforms.ToTensor()
    gt = trans(img).unsqueeze(0)
    kernel1 = de.random_mixed_kernels(kernel_list = kernel_list,
                                      kernel_prob = kernel_prob,
                                      kernel_size = kernel_size,
                                      sigma_x_range = blur_sigma,
                                      sigma_y_range = blur_sigma,
                                      betag_range = betag_range,
                                      betap_range = betap_range,
                                      )
    kernel1 = torch.FloatTensor(kernel1).unsqueeze(0)
    kernel2 = de.random_mixed_kernels(kernel_list = kernel_list,
                                      kernel_prob = kernel_prob,
                                      kernel_size = kernel_size,
                                      sigma_x_range = blur_sigma,
                                      sigma_y_range = blur_sigma,
                                      betag_range = betag_range,
                                      betap_range = betap_range,
                                      )
    kernel2 = torch.FloatTensor(kernel2).unsqueeze(0)
    omega_c = np.random.uniform(np.pi/3, np.pi)
    sinc_kernel = de.circular_lowpass_kernel(omega_c, kernel_size, pad_to=5)
    sinc_kernel = torch.FloatTensor(sinc_kernel).unsqueeze(0)
    
    data = dict(gt=gt, kernel1=kernel1, kernel2=kernel2, sinc_kernel=sinc_kernel)
    _, lq_img = model.feed_data(data)
    lq_img = lq_img.squeeze(0).to("cpu").numpy()
    img_bgr = cv2.cvtColor(np.transpose(lq_img, (1, 2, 0)), cv2.COLOR_RGB2BGR)
    img_bgr = (img_bgr * 255).astype(np.uint8)
    return img_bgr

def gaussian_noise(image, mean=0, var=0.001):
    image = np.array(image/255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out*255)
    return out

def sp_noise(image, prob):
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output

Maker_list = ["GaussianBlur", "GaussianNoise", "Complex"]
with open("../dataset/MyDataset/PromptEval.json", 'r') as file:
    data0 = json.load(file)
out_path_pre = "../dataset/MyDataset/"
img_path_pre = "../dataset/translated-LLaVA-Instruct-150K/llava-imgs/filtered-llava-images"
for kind in Maker_list:
    if kind == "GaussianBlur":
        for it, _data in tqdm(enumerate(data0), total=len(data0)):
            img_path = os.path.join(img_path_pre, _data["image"])
            out_path = os.path.join(out_path_pre, kind, _data["image"])
            image_pil = Image.open(img_path)
            cv2_image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
            out_image = cv2.GaussianBlur(cv2_image, (5, 5), sigmaX=0)
            cv2.imwrite(out_path, out_image)

    elif kind == "GaussianNoise":
        for it, _data in tqdm(enumerate(data0), total=len(data0)):
            img_path = os.path.join(img_path_pre, _data["image"])
            out_path = os.path.join(out_path_pre, kind, _data["image"])
            image_pil = Image.open(img_path)
            cv2_image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
            out_image = gaussian_noise(cv2_image, 0.1, 0.001)
            cv2.imwrite(out_path, out_image)

    elif kind == "Complex":
        for it, _data in tqdm(enumerate(data0), total=len(data0)):
            img_path = os.path.join(img_path_pre, _data["image"])
            out_path = os.path.join(out_path_pre, kind, _data["image"])
            image_pil = Image.open(img_path)
            cv2_image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
            try:
                out_image = complex_degrade(image_pil)
            except RuntimeError:
                out_image = cv2.GaussianBlur(cv2_image, (5, 5), sigmaX=0)
                out_image = gaussian_noise(out_image, 0, 0.01)
                out_image = sp_noise(out_image, random.choice([0.01, 0.03, 0.05]))
            cv2.imwrite(out_path, out_image)


