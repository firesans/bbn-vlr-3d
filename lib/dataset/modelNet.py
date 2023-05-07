from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch_geometric.datasets import ModelNet
import torch_geometric.transforms as T
import torchvision
import numpy as np
# from imbalance_cifar import IMBALANCECIFAR10s
from PIL import Image
import random


class Config:
    def __init__(self, **kwargs):
        for key in kwargs.keys():
            setattr(self, key, kwargs[key])


class IMBALANCEMODELNET10(Dataset):
    def __init__(self, mode, cfg, root='./lib/dataset/ModelNet10', imb_type='exp',
                 transform=None, target_transform=None, download=True):
        super().__init__()
        self.train = True if mode == "train" else False

        self.cfg = cfg
        self.cls_num = cfg.DATASET.CLASSES
        self.transform = T.SamplePoints(2048)
        self.pre_transform = T.NormalizeScale()

        # load the entire dataset into RAM
        if self.train:
            dataset = ModelNet(root=root, name='10', train=True,
                               transform=self.transform, pre_transform=self.pre_transform)
        else:
            dataset = ModelNet(root=root, name='10', train=False,
                               transform=self.transform, pre_transform=self.pre_transform)

        self.data = [dataset[i] for i in range(len(dataset))]
        self.targets = [dataset[i].y for i in range(len(dataset))]

        #! may potentially want to add code for performing data imbalance
        #! so that the data is even more imbalanced, but that is for later

        self.dual_sample = True if cfg.TRAIN.SAMPLER.DUAL_SAMPLER.ENABLE and self.train else False
        print("{} Mode: Contain {} images".format(mode, len(self.data)))
        if self.dual_sample or (self.cfg.TRAIN.SAMPLER.TYPE == "weighted sampler" and self.train):
            self.class_weight, self.sum_weight = self.get_weight(self.get_annotations(), self.cls_num)
            self.class_dict = self._get_class_dict()

        # for i in range(len(self.data)): print(x.data[i].pos.shape, x.data[i].y

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def sample_class_index_by_weight(self):
        rand_number, now_sum = random.random() * self.sum_weight, 0
        for i in range(self.cls_num):
            now_sum += self.class_weight[i]
            if rand_number <= now_sum:
                return i

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.cfg.TRAIN.SAMPLER.TYPE == "weighted sampler" and self.train:
            assert self.cfg.TRAIN.SAMPLER.WEIGHTED_SAMPLER.TYPE in ["balance", "reverse"]
            if self.cfg.TRAIN.SAMPLER.WEIGHTED_SAMPLER.TYPE == "balance":
                sample_class = random.randint(0, self.cls_num - 1)
            elif self.cfg.TRAIN.SAMPLER.WEIGHTED_SAMPLER.TYPE == "reverse":
                sample_class = self.sample_class_index_by_weight()
            sample_indexes = self.class_dict[sample_class]
            index = random.choice(sample_indexes)

        img, target = self.data[index], self.targets[index]
        meta = dict()

        if self.dual_sample:
            if self.cfg.TRAIN.SAMPLER.DUAL_SAMPLER.TYPE == "reverse":
                sample_class = self.sample_class_index_by_weight()
                sample_indexes = self.class_dict[sample_class]
                sample_index = random.choice(sample_indexes)
            elif self.cfg.TRAIN.SAMPLER.DUAL_SAMPLER.TYPE == "balance":
                sample_class = random.randint(0, self.cls_num-1)
                sample_indexes = self.class_dict[sample_class]
                sample_index = random.choice(sample_indexes)
            elif self.cfg.TRAIN.SAMPLER.DUAL_SAMPLER.TYPE == "uniform":
                sample_index = random.randint(0, self.__len__() - 1)
            else:
                print("not given dual sample type")

            sample_img, sample_label = self.data[sample_index], self.targets[sample_index]
            meta['sample_image'] = sample_img.pos
            meta['sample_label'] = sample_label.item()

        return img.pos, target.item(), meta

    def _get_class_dict(self):
        class_dict = dict()
        for i, anno in enumerate(self.get_annotations()):
            cat_id = anno["category_id"]
            if not cat_id in class_dict:
                class_dict[cat_id] = []
            class_dict[cat_id].append(i)
        return class_dict

    def get_weight(self, annotations, num_classes):
        num_list = [0] * num_classes
        cat_list = []
        for anno in annotations:
            category_id = anno["category_id"]
            num_list[category_id] += 1
            cat_list.append(category_id)
        max_num = max(num_list)
        class_weight = [max_num / i for i in num_list]
        sum_weight = sum(class_weight)
        return class_weight, sum_weight

    def get_num_classes(self):
        return self.cls_num

    def reset_epoch(self, epoch):
        self.epoch = epoch

    def get_annotations(self):
        annos = []
        for target in self.targets:
            annos.append({'category_id': int(target)})
        return annos

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list


if __name__ == '__main__':
    # modelnet = ModelNet(root="ModelNet10")
    cfg = Config(
        DATASET=Config(
            CLASSES=10,
            IMBALANCECIFAR=Config(
                RANDOM_SEED=16824,
                RATIO=0.5,
            )
        ),
        TRAIN=Config(
            SAMPLER=Config(
                TYPE="weighted sampler", # "weighted sampler" or "dual sampler"
                DUAL_SAMPLER=Config(
                    TYPE="reverse",  # "reverse", "balance", "uniform"
                    ENABLE=True,
                ),
                WEIGHTED_SAMPLER=Config(
                    TYPE="balance", # "balance", "reverse"
                ),
            )
        )
    )
    ds = IMBALANCEMODELNET10("train", cfg)

    class_dict = ds._get_class_dict()
    annotations = ds.get_annotations()
    count_per_class = {i: len(class_dict[i]) for i in class_dict.keys()}

    dl = DataLoader(ds, batch_size=10)
    out = next(iter(dl))

    import pdb
    pdb.set_trace()

    # trainset = IMBALANCEMODELNET10(root='ModelNet10', train=True)
    # trainloader = iter(trainset)
    # data, label = next(trainloader)
