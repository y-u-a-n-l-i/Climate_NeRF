import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.utils as utils
from .process_stylization import *
from .photo_wct import PhotoWCT
from .photo_gif import GIFSmoothing
from PIL import Image
from mmseg.apis import inference_model, init_model
from mmseg.utils import get_classes

class Stylizer:
    def __init__(self, styl_img_path, kwargs):
        self.p_wct = PhotoWCT()
        self.p_wct.load_state_dict(torch.load("datasets/stylize_tools/PhotoWCTModels/photo_wct.pth"))
        self.p_pro = GIFSmoothing(r=35, eps=0.001)
        self.p_wct.cuda()

        self.styl_img = Image.open(styl_img_path).convert('RGB')
        config_file = kwargs.sem_conf_path
        checkpoint_file = kwargs.sem_ckpt_path
        palette = 'cityscapes'
        sem_model = init_model(config_file, checkpoint=checkpoint_file, device='cuda')
        sem_model.CLASSES = get_classes(palette)
        result = inference_model(sem_model, styl_img_path)
        label = result.pred_sem_seg.data.reshape(-1).cpu()
        label[torch.logical_or(label==0, label==1)] = 0 # road
        label[torch.logical_and(label<=7, label>=2)] = 1
        label[label==8] = 2
        label[label==9] = 3
        label[label==10] = 4
        label[torch.logical_or(label==11, label==12)] = 5
        label[label>=13] = 6
        label = label.reshape(self.styl_img.height, self.styl_img.width)
        self.styl_seg = Image.fromarray(label.numpy().astype(np.uint8))

        self.new_sw, self.new_sh = memory_limit_image_resize(self.styl_img)
        self.styl_seg.resize((self.new_sw, self.new_sh), Image.NEAREST)
        self.styl_seg = np.asarray(self.styl_seg, dtype=np.uint16)

        self.styl_img = transforms.ToTensor()(self.styl_img).unsqueeze(0)
        self.styl_img = self.styl_img.cuda()
    
    def forward(self, rgb, segment):
        '''
        input:
            rgb: torch.tensor
            segment: torch.tensor
        output:
            stylized_rgb: torch.tensor
        '''
        rgb_np = rgb.cpu().numpy()
        rgb_pilimg = Image.fromarray(rgb_np)
        new_cw, new_ch = memory_limit_image_resize(rgb_pilimg)
        cont_pilimg = rgb_pilimg.copy()
        cw = cont_pilimg.width
        ch = cont_pilimg.height

        cont_seg = Image.fromarray(segment.cpu().numpy())
        cont_seg.resize((new_cw, new_ch), Image.NEAREST)

        cont_img = transforms.ToTensor()(rgb_pilimg).unsqueeze(0).cuda()
        cont_seg = np.asarray(cont_seg, dtype=np.uint16)

        stylized_img = self.p_wct.transform(cont_img, self.styl_img, cont_seg, self.styl_seg)
        if ch != new_ch or cw != new_cw:
            stylized_img = nn.functional.upsample(stylized_img, size=(ch, cw), model="bilinear")
        grid = utils.make_grid(stylized_img.data, nrow=1, padding=0)
        ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
        out_img = Image.fromarray(ndarr)

        out_img = self.p_pro.process(out_img, cont_pilimg)
        out_img = smooth_filter(out_img, cont_pilimg, f_radius=15, f_edge=1e-1)

        return transforms.ToTensor()(out_img).reshape(-1, 3)