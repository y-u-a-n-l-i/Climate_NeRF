from datasets.tnt import tntDataset
from .colmap import ColmapDataset
from .tnt import tntDataset
from .kitti360 import KittiDataset
from .mega_nerf.dataset import MegaDataset


dataset_dict = {'colmap': ColmapDataset,
                'tnt': tntDataset,
                'kitti': KittiDataset,
                'mega': MegaDataset
}
