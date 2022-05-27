import sys
import six
import random

# from natsort import natsorted
import PIL
import lmdb
import torch
from torch.utils.data import Dataset, ConcatDataset, sampler
from .rand_aug import Augmentor
import torchvision.transforms as transforms
import traceback


class Batch_Mixed_Dataset(object):

    def __init__(self, opt, dataset_roots, batch_size, learn_type=None):
        self.opt = opt
        if learn_type == 'semi':
            data_type = 'unlabel'
        else:
            data_type = 'label'
        dataloader_iter = None
        dataset_list = []
        for root in dataset_roots:
            if root is None:
                continue
            if data_type == "label":
                cur_dataset = LmdbDataset(root, opt, mode="train")
            else:
                cur_dataset = LmdbDataset_unlabel(root, opt)
            dataset_list.append(cur_dataset)

        train_dataset = ConcatDataset(dataset_list)
        if data_type == "label":
            num_worker = int(opt.workers)
        else:
            num_worker = int(opt.unl_workers)
        data_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=randomSequentialSampler(train_dataset, batch_size),
            num_workers=num_worker,
            pin_memory=False)
        print("%s dataset length: %d * %d" %
              (learn_type, len(data_loader), batch_size))
        self.data_loader = data_loader
        self.dataloader_iter = iter(data_loader)
        return dataloader_iter

    def get_batch(self):
        try:
            data_dict = self.dataloader_iter.next()
        except StopIteration:
            self.dataloader_iter = iter(self.data_loader)
            data_dict = self.dataloader_iter.next()
        except ValueError:
            traceback.print_exc()
            pass
        return data_dict


class randomSequentialSampler(sampler.Sampler):

    def __init__(self, data_source, batch_size):
        self.num_samples = len(data_source)
        self.batch_size = batch_size

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        n_batch = len(self) // self.batch_size
        tail = len(self) % self.batch_size
        index = torch.LongTensor(len(self)).fill_(0)
        for i in range(n_batch):
            random_start = random.randint(0, len(self) - self.batch_size)
            batch_index = random_start + torch.arange(0, self.batch_size)
            index[i * self.batch_size:(i + 1) * self.batch_size] = batch_index
        # deal with tail
        if tail:
            random_start = random.randint(0, len(self) - self.batch_size)
            tail_index = random_start + torch.arange(0, tail)
            index[(i + 1) * self.batch_size:] = tail_index

        return iter(index)


class LmdbDataset(Dataset):

    def __init__(self, root, opt, mode='train'):

        self.root = root
        self.opt = opt
        self.mode = mode
        self.env = lmdb.open(root,
                             max_readers=32,
                             readonly=True,
                             lock=False,
                             readahead=False,
                             meminit=False)
        if not self.env:
            print('cannot open lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            self.nSamples = int(txn.get('num-samples'.encode()))

        self.mode = mode

        Aug = opt.Aug
        self.aug_type = None
        if Aug == 'None' or mode != 'train':
            self.transform = ResizeNormalize((opt.imgW, opt.imgH))
        elif Aug.startswith("rand"):
            self.transform = Rand_augment(opt)
        elif Aug.startswith("weak"):
            self.transform = Weak_augment(opt)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index = index % len(self) + 1

        with self.env.begin(write=False) as txn:
            label_key = 'label-%09d'.encode() % index
            label = txn.get(label_key).decode('utf-8')
            if len(label) > self.opt.batch_max_length:
                return self[index]
            img_key = 'image-%09d'.encode() % index
            imgbuf = txn.get(img_key)
            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)

            try:
                img = PIL.Image.open(buf).convert('RGB')

            except IOError:
                print(f'Corrupted image for {index}')
                # make dummy image and dummy label for corrupted image.
                img = PIL.Image.new('RGB', (self.opt.imgW, self.opt.imgH))
                label = '[dummy_label]'
        img = self.transform(img)
        return_dict = {'img': img, 'label': label}
        return return_dict


class LmdbDataset_unlabel(Dataset):

    def __init__(self, root, opt, mode="train"):

        self.root = root
        self.opt = opt
        self.env = lmdb.open(root,
                             max_readers=32,
                             readonly=True,
                             lock=False,
                             readahead=False,
                             meminit=False)
        if not self.env:
            print('cannot open lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'.encode()))
            self.nSamples = nSamples
        Aug = opt.Aug_semi
        self.weak_transform = Weak_augment(opt)
        if Aug == 'None' or mode != 'train':
            self.transform = ResizeNormalize((opt.imgW, opt.imgH))
        elif Aug.startswith("rand"):
            self.transform = Rand_augment(opt)
        elif Aug.startswith("weak"):
            self.transform = Weak_augment(opt)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index = index % len(self) + 1
        with self.env.begin(write=False) as txn:
            img_key = 'image-%09d'.encode() % index
            imgbuf = txn.get(img_key)
            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)

            try:
                img = PIL.Image.open(buf).convert('RGB')
            except IOError:
                print(f'Corrupted image for {img_key}')
                # make dummy image for corrupted image.
                img = PIL.Image.new('RGB', (self.opt.imgW, self.opt.imgH))
        return_dict = {}
        return_dict['strong_img'] = self.transform(img.copy())
        return_dict['weak_img'] = self.weak_transform(img.copy())
        return return_dict


class ResizeNormalize(object):

    def __init__(self, size, interpolation=PIL.Image.BICUBIC):
        # CAUTION: it should be (width, height). different from size of transforms.Resize (height, width)
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, image):
        image = image.resize(self.size, self.interpolation)
        image = self.toTensor(image)
        image.sub_(0.5).div_(0.5)
        return image


class Weak_augment(object):

    def __init__(self, opt):
        self.opt = opt
        augmentation = []
        augmentation.append(
            transforms.ColorJitter(brightness=0.2,
                                   contrast=0.1,
                                   saturation=0.1,
                                   hue=0.05))

        augmentation.append(
            transforms.Resize((self.opt.imgH, self.opt.imgW),
                              interpolation=PIL.Image.BICUBIC))
        augmentation.append(transforms.ToTensor())
        self.Augment = transforms.Compose(augmentation)

    def __call__(self, image):
        image = self.Augment(image)
        image.sub_(0.5).div_(0.5)

        return image


class Rand_augment(object):

    def __init__(self, opt, use_norm=True):
        self.opt = opt
        augmentation = []
        self.first_augmentor = Augmentor(2, 5, 'spatial')
        self.augmentor = Augmentor(2, 10, 'channel')

        augmentation.append(
            transforms.Resize((self.opt.imgH, self.opt.imgW),
                              interpolation=PIL.Image.BICUBIC))
        augmentation.append(transforms.ToTensor())
        self.normalize = transforms.Compose(augmentation)
        self.use_norm = use_norm

    def __call__(self, image):
        image = self.first_augmentor(image)
        image = self.augmentor(image)
        if self.use_norm:
            image = self.normalize(image)
            image.sub_(0.5).div_(0.5)
        return image
