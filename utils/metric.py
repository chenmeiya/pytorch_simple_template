import torch
import math


def PSNR(img, gt):
    if img.size()==4:
        b, c, w, h = img.shape()
        psnr = 0
        for i in range(b):
            mse = torch.mean((img-gt)**2)
            psnr += 20 * math.log10(1 / math.sqrt(mse.cpu().numpy()))
        psnr = psnr/b
    else:
        mse = torch.mean((img-gt)**2)
        psnr = 20*math.log10(1/math.sqrt(mse.cpu().numpy()))
    return psnr