'''Load image/labels/boxes from an annotation file.

The list file is like:

    img.jpg xmin ymin xmax ymax label xmin ymin xmax ymax label ...
'''
from __future__ import print_function

import os
import sys
import random

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image
from encoder import DataEncoder
from transform import resize, random_flip, random_crop, center_crop


class ListDataset(data.Dataset):
    def __init__(self, root, list_file, train, transform, input_size,is_dubg=False,):
        '''
        Args:
          root: (str) ditectory to images.
          list_file: (str) path to index file.
          train: (boolean) train or test.
          transform: ([transforms]) image transforms.
          input_size: (int) model input size.
        '''
        self.root = root
        self.train = train
        self.transform = transform
        self.input_size = input_size

        self.fnames = []
        self.boxes = []
        self.labels = []

        self.encoder = DataEncoder()
        self.max_objects=50
        self.is_debug=is_dubg
        with open(list_file) as f:
            lines = f.readlines()
            self.num_samples = len(lines)

        for line in lines:
            splited = line.strip().split()
            self.fnames.append(splited[0])
            num_boxes = (len(splited) - 1) // 5
            box = []
            label = []
            for i in range(num_boxes):
                xmin = splited[1+5*i]
                ymin = splited[2+5*i]
                xmax = splited[3+5*i]
                ymax = splited[4+5*i]
                c = splited[5+5*i]
                box.append([float(xmin),float(ymin),float(xmax),float(ymax)])
                label.append(int(c))
            self.boxes.append(torch.Tensor(box))
            self.labels.append(torch.LongTensor(label))

    def __getitem__(self, idx):
        '''Load image.

        Args:
          idx: (int) image index.

        Returns:
          img: (tensor) image tensor.
          loc_targets: (tensor) location targets.
          cls_targets: (tensor) class label targets.
        '''
        # Load image and boxes.
        fname = self.fnames[idx]
        img = Image.open(os.path.join(self.root, fname))
        if img.mode != 'RGB':
            img = img.convert('RGB')

        boxes = self.boxes[idx].clone()
        labels = self.labels[idx]
        size = self.input_size

        # Data augmentation.
        if self.train:
            img, boxes = random_flip(img, boxes)
            # img, boxes = random_crop(img, boxes)
            img, boxes = resize(img, boxes, (size,size))
        else:
           img, boxes = resize(img, boxes, (size,size))
            # img, boxes = center_crop(img, boxes, (size,size))
        img = self.transform(img)

        if self.is_debug:
            filled_labels = np.zeros((self.max_objects, 4), dtype=np.float32)
            if boxes is not None:
                filled_labels[range(len(boxes))[:self.max_objects]] = boxes[:self.max_objects]
            else:
                print('no object')
            filled_labels = torch.from_numpy(filled_labels)
            return img,filled_labels
        else:
            return img, boxes, labels, fname

    def collate_fn(self, batch):
        '''Pad images and encode targets.

        As for images are of different sizes, we need to pad them to the same size.

        Args:
          batch: (list) of images, cls_targets, loc_targets.

        Returns:
          padded images, stacked cls_targets, stacked loc_targets.
        '''
        imgs = [x[0] for x in batch]
        boxes = [x[1] for x in batch]
        labels = [x[2] for x in batch]
        fname=[x[3] for x in batch]


        h = w = self.input_size
        num_imgs = len(imgs)
        inputs = torch.zeros(num_imgs, 3, h, w)

        loc_targets = []
        cls_targets = []
        for i in range(num_imgs):
            inputs[i] = imgs[i]
            loc_target, cls_target = self.encoder.encode(boxes[i], labels[i], input_size=(w,h))
            loc_targets.append(loc_target)
            cls_targets.append(cls_target)
        return inputs, torch.stack(loc_targets), torch.stack(cls_targets),fname

    def __len__(self):
        return self.num_samples


# def test():
#     import torchvision
#
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
#     ])
#     dataset = ListDataset(root="/home/wushuanchen/datasets/xu_tian",
#                           list_file='/home/wushuanchen/PycharmProjects/pytorch-retinanet-master/data/simulate_train.txt', train=True, transform=transform, input_size=600)
#     dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn)
#
#     for images, loc_targets, cls_targets,fnames in dataloader:
#         print(images.size())
#         print(loc_targets.size())
#         print(cls_targets.size())
#         grid = torchvision.utils.make_grid(images, 1)
#         torchvision.utils.save_image(grid, 'a.jpg')
#         break
#
# # test()

if __name__ == "__main__":
    import cv2
    import torchvision
    import numpy as np

    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    dataset = ListDataset(root="/home/wushuanchen/datasets/voc/VOC2012/JPEGImages",
    list_file='/home/wushuanchen/PycharmProjects/pytorch-retinanet-master/data/voc12_train.txt', train=True, transform=transform, input_size=384,is_dubg=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=1)#, collate_fn=dataset.collate_fn)
    for step, (images, loc_targets) in enumerate(dataloader):
        for i, image in enumerate(images):
            image = image.numpy()*255
            image=np.transpose(image,[1,2,0])
            image=np.array(image, np.float32).copy()
            label=loc_targets[i]
            h, w = image.shape[:2]
            for l in label:
                if l.sum() == 0:
                    continue
                x1 = int((l[0]))
                y1 = int((l[1]))
                x2 = int((l[2]))
                y2 = int((l[3]))
                # obj_class=l[4]
                # cv2.putText(image,str(obj_class.item()),(x1,y1),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,0),thickness=2)
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255),thickness=2)
            image = cv2.cvtColor(np.array(image,np.float32), cv2.COLOR_RGB2BGR)
            cv2.imwrite("common/step{}_{}.jpg".format(step, i), image)
        # only one batch
        break