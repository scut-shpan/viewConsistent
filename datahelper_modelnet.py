# coding=utf-8
# load depth picture and shaded picture
import os.path
import numpy as np
import sys
from mxnet.gluon.data.vision.transforms import Compose, RandomResizedCrop, CenterCrop, ToTensor, Normalize
from mxnet.gluon.data.vision import datasets
from mxnet import image, nd

m40_weights = {'laptop': 1.8137266353744221, 'airplane': 1.1240245821485455, 'bathtub': 2.031726749766898,
               'bed': 1.1995863371024584, 'bench': 1.725646702694707, 'bookshelf': 1.158337840451145,
               'bottle': 1.3844767003542713, 'bowl': 2.4038494358929547, 'car': 1.6525141171170625, 'chair': 1.0,
               'cone': 1.7460704530131321, 'cup': 2.2409124522572443, 'curtain': 1.860690862324653,
               'desk': 1.6442098859237197, 'door': 2.01291338469571, 'dresser': 1.6442098859237197,
               'flower_pot': 1.8137266353744221, 'glass_box': 1.7323483105380568, 'guitar': 1.7900150712886589,
               'keyboard': 1.8302535290057265, 'lamp': 1.928235282632091, 'mantel': 1.4628320724724408,
               'monitor': 1.2411271309355467, 'night_stand': 1.6442098859237197, 'person': 2.1617544564237483,
               'piano': 1.5670995781111468, 'plant': 1.547260741696586, 'radio': 2.0446680333909435,
               'range_hood': 1.9772790894490746, 'sink': 1.9079365616548196, 'sofa': 1.0934466946103136,
               'stairs': 1.928235282632091, 'stool': 2.145621359230906, 'table': 1.3138281556027,
               'tent': 1.7602379725202966, 'toilet': 1.3722959221068958, 'tv_stand': 1.493241896265148,
               'vase': 1.2323556213594973, 'wardrobe': 2.170005507613259, 'xbox': 2.0512637824072075}

m10_weights = {'bathtub': 2.0324882666109816, 'bed': 1.1992602388582525, 'chair': 1.0, 'desk': 1.644826156553362,
               'dresser': 1.644826156553362, 'monitor': 1.2415923210581679, 'night_stand': 1.644826156553362,
               'sofa': 1.0938565322403733, 'table': 1.3143205949874932, 'toilet': 1.3714826056081661}



def img_normalization(img):
    img = img.astype('float32') / 255
    normalized_img = image.color_normalize(img, mean=nd.array([0.485, 0.456, 0.406]),
                                           std=nd.array([0.229, 0.224, 0.225]))
    return normalized_img


class MultiViewImageDataset(datasets.ImageFolderDataset):
    def __init__(self, root, num_view, flag=1, transform=None):
        super(MultiViewImageDataset, self).__init__(root, flag, transform)
        self._num_view = num_view
        self.num_classes = len(self.synsets)
        assert self.num_classes in [10, 40]
        if self.num_classes == 40:
            self.weights = m40_weights
        else:
            self.weights = m10_weights

    def __len__(self):
        return len(self.items) // self._num_view

    def __getitem__(self, idx):
        # loader picture and points
        imgs = []
        depths_7 = []
        depths_14 = []
        depths_28 = []
        loader_7 = 1
        loader_14 = 1
        loader_28 = 0
        for item in self.items[idx * self._num_view:idx * self._num_view + self._num_view]:
            listpath = str(item[0]).split('/')
            a = os.path.split(str(item[0]))
            b = os.path.abspath(os.path.join(a[0], '../../..'))
            name = listpath[-1].split('.')[0]
            # get depth_path
            img = image.imread(item[0], self._flag)
            # depth = image.imread(depth_path, self._flag)
            if self._transform is not None:
                img = self._transform(img)
            imgs.append(img.expand_dims(0))
            if loader_7 == 1:
                depth_path_7 = os.path.join(b, 'index_7_36', listpath[-3], listpath[-2], name)
                depth_path_7 = depth_path_7 + '.npy'
                depth_7 = np.load(depth_path_7)
                depth_7 = nd.array(depth_7)
                depths_7.append(depth_7.expand_dims(0))
            if loader_14 == 1:
                depth_path_14 = os.path.join(b, 'index_14_36', listpath[-3], listpath[-2], name)
                depth_path_14 = depth_path_14 + '.npy'
                depth_14 = np.load(depth_path_14)
                depth_14 = nd.array(depth_14)
                depths_14.append(depth_14.expand_dims(0))
            if loader_28 == 1:
                depth_path_28 = os.path.join(b, 'index_28_36', listpath[-3], listpath[-2], name)
                depth_path_28 = depth_path_28 + '.npy'
                depth_28 = np.load(depth_path_28)
                depth_28 = nd.array(depth_28)
                depths_28.append(depth_28.expand_dims(0))
        label = self.items[idx * self._num_view][1]
        ##0
        if loader_7 == 0 and loader_14 == 0 and loader_28 == 0:
            return nd.concat(*imgs, dim=0), label, self.weights[self.synsets[label]]
        ##1
        if loader_7 == 1 and loader_14 == 0 and loader_28 == 0:
            return nd.concat(*imgs, dim=0), nd.concat(*depths_7, dim=0), label, self.weights[self.synsets[label]]

        if loader_7 == 0 and loader_14 == 1 and loader_28 == 0:
            return nd.concat(*imgs, dim=0), nd.concat(*depths_14, dim=0), label, self.weights[self.synsets[label]]

        if loader_7 == 0 and loader_14 == 0 and loader_28 == 1:
            return nd.concat(*imgs, dim=0), nd.concat(*depths_28, dim=0), label, self.weights[self.synsets[label]]
        ##2
        if loader_7 == 1 and loader_14 == 1 and loader_28 == 0:
            return nd.concat(*imgs, dim=0), nd.concat(*depths_7, dim=0), nd.concat(*depths_14, dim=0), label, \
                   self.weights[self.synsets[label]]

        if loader_7 == 1 and loader_14 == 0 and loader_28 == 1:
            return nd.concat(*imgs, dim=0), nd.concat(*depths_7, dim=0), nd.concat(*depths_28, dim=0), label, \
                   self.weights[self.synsets[label]]

        if loader_7 == 0 and loader_14 == 1 and loader_28 == 1:
            return nd.concat(*imgs, dim=0), nd.concat(*depths_14, dim=0), nd.concat(*depths_28, dim=0), label, \
                   self.weights[self.synsets[label]]
        ##3
        if loader_7 == 1 and loader_14 == 1 and loader_28 == 1:
            return nd.concat(*imgs, dim=0), nd.concat(*depths_7, dim=0), nd.concat(*depths_14, dim=0), nd.concat(
                *depths_28, dim=0), label, self.weights[self.synsets[label]]



if __name__ == '__main__':
    train_ds = MultiViewImageDataset(os.path.join(
        '/home/shpan/Downloads/shpan/experiment/ModelNet_Blender_OFF2Multiview-master/trash/mydata10-1/shaded',
        'train'), 12, transform=Compose([
        # RandomResizedCrop(size=(112, 112), scale=(0.5, 1.0), ratio=(1. - 0.1, 1. + 0.1)),
        ToTensor(),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))]))
    print(train_ds.items)
