import os
import cv2
import torch
import numpy as np
from craft import CRAFT
import craft_utils
import imgproc
import file_utils
from torch.autograd import Variable
from collections import OrderedDict

# Yolu buradan ver
image_path = './test_images/example.jpg'
model_path = './craft_mlt_25k.pth'

# Cihaz ayarı
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# module. önekini kaldırmak için
def copyStateDict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("module.", "") if k.startswith("module.") else k
        new_state_dict[name] = v
    return new_state_dict

# Modeli yükle
net = CRAFT()
net.load_state_dict(copyStateDict(torch.load(model_path, map_location=device)))
net.to(device).eval()

# Görüntüyü yükle
image = imgproc.loadImage(image_path)

# Resize ve normalize
img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, 1280, interpolation=cv2.INTER_LINEAR, mag_ratio=1.5)
ratio_h = ratio_w = 1 / target_ratio
x = imgproc.normalizeMeanVariance(img_resized)
x = torch.from_numpy(x).permute(2, 0, 1)
x = Variable(x.unsqueeze(0)).to(device)

# Tahmin yap
with torch.no_grad():
    y, _ = net(x)

score_text = y[0, :, :, 0].cpu().data.numpy()
score_link = y[0, :, :, 1].cpu().data.numpy()

# Kutu çıkarımı
boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold=0.7, link_threshold=0.4, low_text=0.4)
boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)

# Sonucu kaydet
result_folder = './result/'
os.makedirs(result_folder, exist_ok=True)
file_utils.saveResult(image_path, image[:, :, ::-1], boxes, dirname=result_folder)

print("✅ Metin tespiti tamamlandı. Sonuç ./result klasörüne kaydedildi.")
