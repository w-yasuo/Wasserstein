import os
import cv2
import torch
import random
import shutil
import numpy as np
from PIL import Image
from collections import defaultdict
from torchvision import transforms
from torchvision.datasets.vision import VisionDataset

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


class SelfDatasetFolder(VisionDataset):

    def __init__(self, imgroot, transform=None):
        super(SelfDatasetFolder, self).__init__(imgroot, transform=transform)
        samples = self.make_dataset(imgroot)
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"
                                                                                 "Supported extensions are: " + ",".join(
                IMG_EXTENSIONS)))

        self.samples = samples
        # self.targets = [s[1] for s in samples]

    def _find_classes(self, dir):
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        idx_to_class = {i: classes[i] for i in range(len(classes))}
        return classes, class_to_idx, idx_to_class

    def make_dataset(self, imgroot):
        instances = []
        for r, dir, files in os.walk(imgroot):
            for file in files:
                fn, ext = os.path.splitext(file)
                if ext not in IMG_EXTENSIONS:
                    continue

                if os.path.exists(os.path.join(r, file)):
                    img = cv2.imread(os.path.join(r, file))
                    hist1 = cv2.calcHist([img], [0], None, [256], [0, 256])
                    hist2 = cv2.calcHist([img], [1], None, [256], [0, 256])
                    hist3 = cv2.calcHist([img], [2], None, [256], [0, 256])
                    target = np.concatenate((hist1, hist2, hist3))
                    target = target.reshape((3, 256, 1))

                item = os.path.join(r, file), target
                instances.append(item)
        return instances

    def loader(self, path):
        try:
            with open(path, 'rb') as f:
                img = Image.open(f)
                return img.convert('RGB')
        except:
            print("Readimg error !!!")

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, path, target

    def __len__(self):
        return len(self.samples)


if __name__ == "__main__":
    imgroot = r'D:\Datasets\yunse\val'
    dataset = SelfDatasetFolder(imgroot)
    print("data num: ", len(dataset))
    sample, path = dataset[0]
    print(sample, path)
    sample, path = dataset[-1]
    print(sample, path)
    # instances = []
    # for r, dir, files in os.walk(r'D:\Datasets\yunse\val'):
    #     print(files)
    #     for file in files:
    #         #     print(file)
    #         #     fn, ext = os.path.splitext(file)
    #         #     if ext not in IMG_EXTENSIONS:
    #         #         continue
    #         item = os.path.join(r, file)
    #         instances.append(item)
