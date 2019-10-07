from torch.utils import data
import torchvision.transforms as transforms
import os
import glob
import numpy as np
import random
from PIL import Image

transform_train = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop([224, 224]),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    transforms.ToTensor(),
    transforms.Normalize((0.2324, 0.2721, 0.3448), (0.1987, 0.2172, 0.2403))])


transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.2324, 0.2721, 0.3448), (0.1987, 0.2172, 0.2403))
])

class VideoDataset(data.Dataset):
    def __init__(self, source, transform, classes):
        self.source = source
        self.transform = transform
        self.classez = classes
        self.keys = []
        self.segments = 2

        #Load Meta_data [(video_name, class), ...]
        for cls in self.classez:
            videos = os.listdir(os.path.join(self.source, cls))
            for vid in videos:
                self.keys.append((vid, cls))

        print('Found ', len(self.keys), ' videos')

    def _pil_loader(self, path):
        try:
            with open(path, 'rb') as f:
                img = Image.open(f)
                return img.convert('RGB')
        except OSError as e:
            return Image.new('RGB', (256, 256))

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        video, cls = self.keys[idx]
        selector = os.path.join(self.source, cls, video, '*.jpg')
        jpg_files = glob.glob(selector)
        ls = np.linspace(0, len(jpg_files)-1, num=self.segments+1)
        ls = ls.astype(int)

        # This can be rewriten in a for loop according to the size of self.segments
        # Written like this for the sake of clarity
        frame_0 = random.randint(ls[0], ls[1])
        frame_1 = random.randint(ls[1], ls[2])

        img0 = self.transform(self._pil_loader(jpg_files[frame_0]))
        img1 = self.transform(self._pil_loader(jpg_files[frame_1]))

        return img0, img1, self.classez.index(cls)


# Quick and dirty fix, since TSN sampling is sparse repeat evauation set 10 times, dont be like me ;)
class VideoDatasetVal(VideoDataset):
    def __len__(self):
        return len(self.keys)*10

    def __getitem__(self, idx):
        video, cls = self.keys[idx%len(self.keys)]
        selector = os.path.join(self.source, cls, video, '*.jpg')
        jpg_files = glob.glob(selector)
        ls = np.linspace(0, len(jpg_files)-1, num=self.segments+1)
        ls = ls.astype(int)

        frame_0 = random.randint(ls[0], ls[1])
        frame_1 = random.randint(ls[1], ls[2])

        img0 = self.transform(self._pil_loader(jpg_files[frame_0]))
        img1 = self.transform(self._pil_loader(jpg_files[frame_1]))

        return img0, img1, self.classez.index(cls)
