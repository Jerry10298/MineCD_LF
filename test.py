import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
from torch.utils.data import DataLoader
import logging
import warnings
from LFDatatest_syn import LFDataset
from Functions import *
import argparse
import lpips
from skimage.metrics import structural_similarity
import numpy as np
# from DCUNet import Main_unrolling as Net
# from model.ISRNet import get_model as Net
from model.laformer_lf import get_model as Net
# from model.Net import get_model as Net
# from model.EPIT import get_model as Net
# from bioir_lf import get_model as Net

# from Ablation.wo_mamba import get_model as Net
# from model.Net_newpr import get_model as Net
import re
# from OursNet import MainSAST_unrolling as Net
# from LFCID.CIDNet import CIDNet as Net
# from mamba.MLFSR_w_CPE import get_model as Net
# from mamba_ICUS.MLFSR import get_model as Net
# from Walmafa.Walmafa import Walmafa as Net
# from IAT.IAT_main import IAT as Net
# from PFINet import Net
# from HVI.CIDNet import CIDNet as Net
def cal_ssim(img1, img2):
    img1_np = img1.data.cpu().numpy()
    img2_np = img2.data.cpu().numpy()

    ssim = 0
    for i in range(img1.shape[0]):
        ssim += structural_similarity(img1_np[i, ...], img2_np[i, ...], gaussian_weights=True, data_range=1.0)
    ssim = ssim/img1.shape[0]
    return ssim

parser = argparse.ArgumentParser(description="AUNS-LL")
parser.add_argument("--stageNum", type=int, default=3, help="The number of stages")
parser.add_argument("--sasLayerNum", type=int, default=6, help="The number of layers")
parser.add_argument("--batchSize", type=int, default=1, help="Batch size")
parser.add_argument("--patchSize", type=int, default=256, help="The size of croped LF patch")
parser.add_argument("--angResolution", type=int, default=5, help="The angular resolution of original LF")
parser.add_argument("--overlap", type=int, default=4, help="The size of croped LF patch")
parser.add_argument("--summaryPath", type=str, default='./', help="Path for saving training log ")
parser.add_argument("--channels", type=int, default=64, help="channels")
# parser.add_argument("--scale_factor", type=int, default=2, help="4, 2")
parser.add_argument("--dataName", type=str, default='Synthetic', help="The name of dataset ")
parser.add_argument("--modelPath", type=str, default='./model_pth/laformer_lf_wo_mohu_mix2_Synthetic_10010_0.0001/', help="Path for loading trained model ")
# parser.add_argument("--modelPath", type=str, default='./model/lfll_unrolling_Synthetic_10010_0.0001/', help="Path for loading trained model ")#lfll_unrolling_Synthetic_10010_0.0001
parser.add_argument("--dataPath", type=str, default=r'./Test/test_mix2.mat', help="Path for test data ")
parser.add_argument("--savePath", type=str, default='./results/', help="Path for saving results ")
parser.add_argument("--angular_in", type=int, default=5, help="angular resolution")
parser.add_argument("--angular_out", type=int, default=5, help="angular resolution")
parser.add_argument("--factor", type=int, default=2, help="upscale factor")
parser.add_argument('--cora_sec_score_mode', type=str,
    default='full',
    choices=[
        'sim_only',
        'sim_rel',
        'sim_deg',
        'full'
    ]
)

opt = parser.parse_known_args()[0]

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
log = logging.getLogger()
fh = logging.FileHandler('ISRNet_wo_mohu_mix2__test.log')
log.addHandler(fh)

lf_dataset = LFDataset(opt)
dataloader = DataLoader(lf_dataset, batch_size=opt.batchSize,shuffle=False)
print('loaded {} LFIs from {}'.format(len(dataloader), opt.dataPath))

loss_lpips_alex = lpips.LPIPS(net='alex')
model=Net(opt)
# model=Net()

ckpts = []

# 先收集所有 checkpoint： (epoch, path)
for root, dirs, files in os.walk(opt.modelPath):
    for file in files:
        # 只匹配你这种命名：model_epoch_XXXX.pth
        m = re.match(r'model_epoch_(\d+)\.pth$', file)
        if not m:
            continue

        epoch = int(m.group(1))          # '0100' -> 100, '10000' -> 10000
        path = os.path.join(root, file)
        ckpts.append((epoch, path))

# 按 epoch 数字排序
ckpts.sort(key=lambda x: x[0])

for e, p in ckpts:
    print("找到模型: epoch =", e, "path =", p)

for epoch, path in ckpts:
    if epoch <5000:
        continue

    log.info(f"Test epoch {epoch}: {path}")

    state = torch.load(path)
    model.load_state_dict(state['model'])
    model.eval()
    model.cuda()

    with torch.no_grad():
        num = 0
        avg_psnr = 0
        avg_ssim = 0
        avg_lpips = 0
        avg_time = 0
        avg_gmsd = 0
        avg_mdsi = 0
        for _, sample in enumerate(dataloader):
            num += 1
            LF = sample['lf']          # [b,u,v,c,x,y]
            lowlf = sample['lowlf'].cuda()
            b, u, v, c, x, y = LF.shape

            LFStack, coordinate = CropLF(lowlf, opt.patchSize, opt.overlap)
            n = LFStack.shape[1]
            estiLFStack = torch.zeros(b, n, u, v, c,
                                      opt.patchSize, opt.patchSize).cuda()
            for i in range(n):
                estiLFStack[:, i] = model(LFStack[:, i].cuda())

            estiLF = MergeLF(estiLFStack, coordinate, opt.overlap, x, y)
            b, u, v, c, xCrop, yCrop = estiLF.shape
            estiLF = torch.clamp(estiLF, 0.0, 1.0)
            LF = LF[:, :, :, :,
                    opt.overlap // 2:opt.overlap // 2 + xCrop,
                    opt.overlap // 2:opt.overlap // 2 + yCrop]

            lf_psnr = lf_ssim = lf_lpips = lf_uqi = lf_mdsi = 0

            for ind_uv in range(u * v):
                esti_view = estiLF.reshape(b, u * v, c, xCrop, yCrop)[0, ind_uv].cpu().numpy()
                gt_view   = LF.reshape(b, u * v, c, xCrop, yCrop)[0, ind_uv].cpu().numpy()

                lf_psnr += ComptPSNR(esti_view, gt_view) / (u * v)

                lf_ssim += structural_similarity(
                    (estiLF.permute(0, 1, 2, 4, 5, 3)
                     .reshape(b, u * v, xCrop, yCrop, c)[0, ind_uv].cpu().numpy() * 255).astype(np.uint8),
                    (LF.permute(0, 1, 2, 4, 5, 3)
                     .reshape(b, u * v, xCrop, yCrop, c)[0, ind_uv].cpu().numpy() * 255).astype(np.uint8),
                    gaussian_weights=True, sigma=1.5, use_sample_covariance=False,
                    multichannel=True, win_size=7, channel_axis=-1
                ) / (u * v)

                lf_lpips += loss_lpips_alex(
                    estiLF.reshape(b, u * v, c, xCrop, yCrop)[0, ind_uv].cpu(),
                    LF.reshape(b, u * v, c, xCrop, yCrop)[0, ind_uv]
                ) / (u * v)

                lf_uqi  += uqi(esti_view,  gt_view)  / (u * v)
                lf_mdsi += mdsi(esti_view, gt_view) / (u * v)

            avg_psnr += lf_psnr / len(dataloader)
            avg_ssim += lf_ssim / len(dataloader)
            avg_lpips += lf_lpips / len(dataloader)
            avg_gmsd += lf_uqi / len(dataloader)
            avg_mdsi += lf_mdsi / len(dataloader)

            log.info('Index: %d  PSNR: %.2f  SSIM: %.3f   LPIPS: %.3F  uqi: %.3f MDSI: %.3f'
                     % (num, lf_psnr, lf_ssim, lf_lpips, lf_uqi, lf_mdsi))

    log.info('Epoch %d Average PSNR: %.2f  SSIM: %.3f  LPIPS: %.3f uqi: %.3f MDSI: %.3f'
             % (epoch, avg_psnr, avg_ssim, avg_lpips, avg_gmsd, avg_mdsi))
