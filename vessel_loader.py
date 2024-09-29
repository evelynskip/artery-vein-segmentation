import os
import torch
import numpy as np
import imageio.v2 as imageio
from PIL import Image
import torchio as tio
import matplotlib.pyplot as plt
from torch.utils import data
from utils.augmentations import *
from os.path import splitext,join

def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def worker_init_fn(worker_id):
    np.random.seed(0 + worker_id)

class VesselDataSequence(data.Dataset):
    """support 2 imgs between large interval
       Return (fc-3) (fc-2) fc-1 fc
    """
    colors = [  [  0,   0,   0],
        [0, 255, 0],
        [0, 0, 255]
    ]

    label_colours = dict(zip(range(3), colors))

    def __init__(
        self,
        root,
        split="train",
        augmentations=False,
        test_mode=False,
        path_num=2,
        interval=1,
        force=False
    ):
        """__init__

        :param root:
        :param split:
        :param is_transform:
        :param img_size:
        :param augmentations
        """
        self.path_num= path_num
        self.root = root
        self.split = split
        self.aug_flag = augmentations
        self.test_mode=test_mode
        self.n_classes = 3
        self.files = {}
        self.force = force

        self.images_base = os.path.join(self.root, "Videos_seq",self.split)
        self.videos_base = os.path.join(self.root, "Videos",self.split)
        self.annotations_base = os.path.join(self.root, "Annotations")
        self.forces_base = os.path.join(self.root, "Forces")

        self.files[split] = os.listdir(self.images_base)

        self.files[split].sort() 
        self.void_classes = []
        self.valid_classes = [0,1,2]
        self.class_names = [
            "background",
            "vein",
            "artery",
        ]

        self.ignore_index = 250 
        self.class_map = dict(zip(self.valid_classes, range(3)))

        if not self.files[split]:
            raise Exception("No files for split=[%s] found in %s" % (split, self.images_base))
        print("Found %d %s images" % (len(self.files[split]), split))

    def __len__(self):
        """__len__"""
        return len(self.files[self.split])

    def __getitem__(self, index):
        """__getitem__

        :param index:
        """
        if not self.test_mode:
            img_name= self.files[self.split][index].rstrip()
            lbl_path = os.path.join(
                self.annotations_base,
                img_name,
            )
            lbl = imageio.imread(lbl_path)
 
            id_pos = img_name.rfind('_')
            cur_frame,vid_id = img_name[id_pos+1:-4] , img_name[:id_pos]
            f4_id = int(cur_frame)

            # Get images
            f4_path = os.path.join(self.videos_base, img_name)
            f4_img = imageio.imread(f4_path)

            f3_id = f4_id - self.interval
            f3_path = os.path.join(self.videos_base, ("%s_%s.png" % (vid_id, str(f3_id).zfill(3))))
            f3_img = imageio.imread(f3_path)
            img_sequences = [f4_img,f3_img]

            if self.path_num > 2:           
                f2_id = f4_id - 2
                f2_path = os.path.join(self.videos_base, ("%s_%s.png" % (vid_id, str(f2_id).zfill(3))))
                f2_img = imageio.imread(f2_path)
                img_sequences.append(f2_img)
                if self.path_num > 3:
                    f1_id = f4_id - 3
                    f1_path = os.path.join(self.videos_base, ("%s_%s.png" % (vid_id, str(f1_id).zfill(3))))
                    f1_img = imageio.imread(f1_path)
                    img_sequences.append(f1_img)
            
            # Augmentation
            if self.aug_flag:
                self.augmentations = Compose([ColorJitter([0.6,0.6,0.6]),
                                RandomTranslate((50,50)),
                                RandomRotate(10),
                                RandomVerticallyFlip(0.5),
                                RandomHorizontallyFlip(0.5),
                                Scale((256,256))])                    
                self.tio_augmentations = tio.Compose(
                        [RandomBiasField(coefficients_range=[0,0.5],order=2),#0.3
                        RandomNoise(mean_range=(-1.5,1.5),std_range=(-0.05,0.05))])#1,0.03
            else:
                 self.augmentations = Compose([Scale((256,256))])

            img_sequences,lbl = self.augmentations(img_sequences,lbl) #images: PIL Image


            # To tensor
            lbl = np.array(lbl,dtype=np.uint8)
            lbl = torch.from_numpy(lbl).long()
            img_sequences = self.transfer(img_sequences,aug=self.aug_flag)

            # Torchio tranformation
            if self.aug_flag:
                img_sequences = self.tio_augmentations(img_sequences)
                for i in range(len(img_sequences)):
                    img_sequences[i] = img_sequences[i].squeeze(3)

            # Reverse and divided by 255
            img_sequences = [img/255 for img in img_sequences[::-1]] 
            # Normalization
            self.norm = transforms.Normalize(mean=[0.28, 0.28, 0.28],std=[0.17, 0.17, 0.17])
            img_sequences = [self.norm(img) for img in img_sequences]

            # Get force
            if self.force:
                force_file = os.path.join(self.forces_base,vid_id+'.txt')
                if self.path_num == 2:
                    force_data = np.genfromtxt(force_file,dtype=float)[f4_id-self.interval-1:f4_id,:]
                else:
                    force_data = np.genfromtxt(force_file,dtype=float)[f4_id-self.path_num:f4_id,:]
                force = torch.tensor(force_data)
                return img_sequences,lbl,force
            else:
                return img_sequences,lbl # [f1,f2,f3,f4]   


    def transfer(self,img_sequences,aug):
        # Transfer from PIL.Image to Tensor[C H W]
        img_sequences_transfered = []
        for img in img_sequences:
            img = np.array(img, dtype=np.uint8)
            img = torch.from_numpy(img).float()
            img = img.permute(2,0,1)# c h w
            if aug:
                img = img.unsqueeze(3)
            img_sequences_transfered.append(img)
        return img_sequences_transfered


    def decode_segmap(self, temp):
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = self.label_colours[l][0]
            g[temp == l] = self.label_colours[l][1]
            b[temp == l] = self.label_colours[l][2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb

    def encode_segmap(self, mask):
        # Put all void classes to zero
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask

    def decode_pred(self, mask):
        # Put all void classes to zero
        for _predc in range(self.n_classes):
            mask[mask == _predc] = self.valid_classes[_predc]
        return mask.astype(np.uint8)

class VesselDataforce(data.Dataset):
    """"
    if path_num==2 return max.png,cur_frame,lbl,(force)
    if path_num==3 return min.png,max.png,cur_frame,lbl,(force)
    lbl: Tensor of shape BxHxW
    force: BxTx6 (f_min) f_max f_cur
    """
    colors = [  [  0,   0,   0],
        [0, 255, 0],
        [0, 0, 255]
    ]

    label_colours = dict(zip(range(3), colors))

    def __init__(
        self,
        root,
        split="train",
        augmentations=None,
        test_mode=False,
        path_num=2,
        force=False,
    ):
        """__init__

        :param root:
        :param split:
        :param is_transform:
        :param img_size:
        :param augmentations
        """
        self.path_num= path_num
        self.root = root
        self.split = split
        self.aug_flag = augmentations
        self.test_mode=test_mode
        self.n_classes = 3
        self.files = {}
        self.force = force

        self.images_base = os.path.join(self.root, "Videos",self.split)
        self.videos_base = os.path.join(self.root, "Force_img")
        self.annotations_base = os.path.join(self.root, "Annotations")
        self.forces_base = os.path.join(self.root, "Forces")

        self.files[split] = os.listdir(self.images_base)
        self.files[split].sort()
        self.void_classes = []
        self.valid_classes = [0,1,2]
        self.class_names = [
            "background",
            "vein",
            "artery",
        ]

        self.ignore_index = 250 
        self.class_map = dict(zip(self.valid_classes, range(3)))

        if not self.files[split]:
            raise Exception("No files for split=[%s] found in %s" % (split, self.images_base))

        print("Found %d %s images" % (len(self.files[split]), split))

    def __len__(self):
        """__len__"""
        return len(self.files[self.split])

    def __getitem__(self, index):
        """__getitem__

        :param index:
        """
        if not self.test_mode:
            img_name= self.files[self.split][index].rstrip()
            lbl_path = os.path.join(
                self.annotations_base,
                img_name,
            )
            lbl = imageio.imread(lbl_path)
            # lbl = self.encode_segmap(np.array(lbl, dtype=np.uint8)) # no need 
            # get videoid(include clipid) and cur frameid
            id_pos = img_name.rfind('_')
            cur_frame,vid_id = img_name[id_pos+1:-4] , img_name[:id_pos]
            f4_id = int(cur_frame)

            f4_path = os.path.join(self.images_base, img_name)
            f4_img = imageio.imread(f4_path)
            
            f_max_path = os.path.join(self.videos_base, ("%s_max.png" % (vid_id)))
            f_min_path = os.path.join(self.videos_base, ("%s_min.png" % (vid_id)))
            img_sequences = [f4_img]

            if self.path_num >= 2:
                f_max_img = imageio.imread(f_max_path)
                img_sequences.append(f_max_img)
                if self.path_num >= 3:
                    f_min_img = imageio.imread(f_min_path)
                    img_sequences.append(f_min_img)
            # Augmentation
            height,width = f4_img.shape[0],f4_img.shape[1]
            if self.aug_flag:
                self.augmentations = Compose([ColorJitter([0.6,0.6,0.6]),
                                RandomTranslate((50,50)),
                                RandomRotate(10),
                                RandomVerticallyFlip(0.5),
                                RandomHorizontallyFlip(0.5),
                                Scale((256,256))])                    
                self.tio_augmentations = tio.Compose(
                        [RandomBiasField(coefficients_range=[0,0.5],order=2),
                        RandomNoise(mean_range=(-1.5,1.5),std_range=(-0.05,0.05))])
            else:
                 self.augmentations = Compose([Scale((256,256))])
            img_sequences,lbl = self.augmentations(img_sequences,lbl) # <class 'PIL.Image.Image'>
                
            # To tensor
            lbl = np.array(lbl,dtype=np.uint8)
            lbl = torch.from_numpy(lbl).long()
            img_sequences = self.transfer(img_sequences,aug=self.aug_flag) # <class 'torch.Tensor'> torch.Size([3, 300, 200, 1])


           # Torchio tranformation
            if self.aug_flag:
                img_sequences = self.tio_augmentations(img_sequences) # input:[C H W 1] output:<class 'torch.Tensor'> torch.Size([3, 300, 200, 1])
                for i in range(len(img_sequences)):
                    img_sequences[i] = img_sequences[i].squeeze(3)

            # Reverse and divided by 255
            img_sequences = [img/255 for img in img_sequences[::-1]] 
            # Normalization
            self.norm = transforms.Normalize(mean=[0.28, 0.28, 0.28],std=[0.17, 0.17, 0.17])
            img_sequences = [self.norm(img) for img in img_sequences]

            # get force
            if self.force:
                force_file = os.path.join(self.forces_base,vid_id+'.txt') 
                force_data = np.genfromtxt(force_file,dtype=float)
                f_cur = force_data[f4_id-1,:]
                f_max = force_data[np.argmin(force_data[:,2]),:]
                f_min = force_data[np.argmax(force_data[:,2]),:]
                if self.path_num == 2:
                    force = torch.tensor(np.vstack((f_max, f_cur)))
                elif self.path_num == 3:
                    force = torch.tensor(np.vstack((f_min, f_max, f_cur)))
                return img_sequences,lbl,force # [f_min_img/255, f_max_img/255, f4_img/255],lbl,force
            else:   
                return img_sequences,lbl
                
    def transfer(self,img_sequences,aug):
        # Transfer from PIL.Image to Tensor[C H W]
        img_sequences_transfered = []
        for img in img_sequences:
            img = np.array(img, dtype=np.uint8)
            img = torch.from_numpy(img).float()
            img = img.permute(2,0,1)# c h w
            if aug:
                img = img.unsqueeze(3)
            img_sequences_transfered.append(img)
        return img_sequences_transfered
    
    def decode_segmap(self, temp):
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = self.label_colours[l][0]
            g[temp == l] = self.label_colours[l][1]
            b[temp == l] = self.label_colours[l][2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb

    def encode_segmap(self, mask):
        # Put all void classes to zero
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask

    def decode_pred(self, mask):
        # Put all void classes to zero
        for _predc in range(self.n_classes):
            mask[mask == _predc] = self.valid_classes[_predc]
        return mask.astype(np.uint8)