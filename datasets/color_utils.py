import cv2
from einops import rearrange
import imageio
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from .shadow_tools.MTMT.networks.MTMT import build_model
from .shadow_tools.MTMT.utils.util import crf_refine

def srgb_to_linear(img):
    limit = 0.04045
    return np.where(img>limit, ((img+0.055)/1.055)**2.4, img/12.92)


def linear_to_srgb(img):
    limit = 0.0031308
    img = np.where(img>limit, 1.055*img**(1/2.4)-0.055, 12.92*img)
    img[img>1] = 1 # "clamp" tonemapper
    return img


def read_image(img_path, img_wh):
    img = imageio.imread(img_path).astype(np.float32)/255.0
    # img[..., :3] = srgb_to_linear(img[..., :3])
    if img.shape[2] == 4: # blend A to RGB
        img = img[..., :3]*img[..., -1:]+(1-img[..., -1:])

    img = cv2.resize(img, img_wh)
    img = rearrange(img, 'h w c -> (h w) c')

    return img

def read_normal_up(img_path, img_wh):
    img = imageio.imread(img_path).astype(np.float32)/255.0

    img = cv2.resize(img, img_wh)
    img = rearrange(img, 'h w -> (h w)')
    
    mask = img>0
    img[mask] = 1
    # import ipdb; ipdb.set_trace()
    return img

def read_normal(norm_path, norm_wh):
    norm = imageio.imread(norm_path).astype(np.float32)/255.0
    # img[..., :3] = srgb_to_linear(img[..., :3])
    if norm.shape[2] == 4: # blend A to RGB
        norm = norm[..., :3]*norm[..., -1:]+(1-norm[..., -1:])

    norm = cv2.resize(norm, norm_wh)
    norm = rearrange(norm, 'h w c -> (h w) c')
    
    norm = np.array(norm)+1e-6
    norm = norm*2.-1.
    norm[:, 1:] = -norm[:, 1:]
    norm = norm/np.linalg.norm(norm, ord=2, axis=-1, keepdims=True)
    
    up = np.zeros_like(norm)
    up[..., 1] = 1
    angle = np.matmul(up.reshape(-1, 3)[:, None, :], norm.reshape(-1, 3)[:, :, None]).squeeze(-1)\
            /(np.linalg.norm(up.reshape(-1, 3), axis=-1, keepdims=True)*np.linalg.norm(norm.reshape(-1, 3), axis=-1, keepdims=True)+1e-6)
    mask = angle>.5
    norm_up = np.zeros_like(angle)
    norm_up[mask] = 1
    
    return norm, norm_up.reshape(-1)

def read_semantic(sem_path, sem_wh, classes=7):
    label = imageio.imread(sem_path).astype(np.uint64)

    label = cv2.resize(label, sem_wh)
    label = rearrange(label, 'h w -> (h w)')
    
    # import ipdb; ipdb.set_trace()
    return label

class Shadow_predictor:
    def __init__(self, ckpt_path):
        self.net = build_model('resnext101').cuda()
        self.net.load_state_dict(torch.load(ckpt_path))
        print("init shadow predictor from {}".format(ckpt_path))
        self.net.eval()
        normal = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.trans_scale = 416
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trans_scale, self.trans_scale)),
            transforms.ToTensor(),
            normal
        ])
        self.to_pil = transforms.ToPILImage()

    def predict(self, img_path):
        img = Image.open(img_path).convert('RGB')
        w, h = img.size
        img_var = self.img_transform(img).unsqueeze(0).cuda()
        _, _, _, up_shadow_final = self.net(img_var)
        res = torch.sigmoid(up_shadow_final[-1])
        prediction = np.array(transforms.Resize((h, w))(self.to_pil(res.data.squeeze(0).cpu())))
        prediction = crf_refine(np.array(img.convert('RGB')), prediction)
        prediction = np.array(self.to_pil(res.data.squeeze(0).cpu()))
        prediction = crf_refine(np.array(img.convert('RGB').resize((self.trans_scale, self.trans_scale))), prediction)
        prediction = np.array(transforms.Resize((h, w))(Image.fromarray(prediction.astype('uint8')).convert('L')))

        return rearrange(torch.FloatTensor(prediction>=(255/2)), 'h w -> (h w)')