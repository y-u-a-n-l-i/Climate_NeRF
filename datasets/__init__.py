from datasets.tnt import tntDataset
from .nsvf import NSVFDataset
from .colmap import ColmapDataset
from .tnt import tntDataset
from .nerf import NeRFDataset
from .nerfpp import NeRFPPDataset
from .kitti360 import KittiDataset
from .mega_nerf.dataset import MegaDataset


dataset_dict = {'nerf': NeRFDataset,
                'nsvf': NSVFDataset,
                'colmap': ColmapDataset,
                'nerfpp': NeRFPPDataset,
                'tnt': tntDataset,
                'nerfpp': NeRFPPDataset,
                'kitti': KittiDataset,
                'mega': MegaDataset
}
