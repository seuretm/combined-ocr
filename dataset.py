import json
from PIL import Image
import os
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

cs = json.load(open('charmap.json', 'rt'))
cs['PAD'] = len(cs)+1

def ocr_collate_batch(batch):
    im_list = []
    l_list = []
    for (im,l) in batch:
        im_list.append(im.permute(2, 0, 1))
        l_list.append(l)
    base_im_width = [im.shape[0] for im in im_list]
    base_txt_length = [len(txt) for txt in l_list]
    im_list = pad_sequence(im_list, batch_first=False, padding_value=0).permute(1, 2, 3, 0)
    im_list = torch.stack([x for x in im_list])
    if type(l)!=str:
        l_list = pad_sequence(l_list, batch_first=True, padding_value=cs['PAD'])
    return im_list, base_im_width, l_list, base_txt_length

class OCRDataset(Dataset):
    def __init__(self, folder, charset, transform=None, height=32, cache=True):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.base_names = [os.path.join(folder, x.replace('.txt', '')) for x in os.listdir(folder) if x.endswith('.txt')]
        self.transform = transform
        self.height = height
        self.charset = charset
        self.cache = {} if cache else None

    def __len__(self):
        return len(self.base_names)

    def __getitem__(self, idx):
        if self.cache is not None and idx in self.cache:
            im, text = self.cache[idx]
            if self.transform:
                im = self.transform(im)
            return im, text
        im = Image.open('%s.jpg' % self.base_names[idx]).convert('RGB')
        if im.size[1]!=self.height:
            ratio = self.height / im.size[1]
            width = int(im.size[0] * ratio)
            try:
                im = im.resize((width,self.height), Image.Resampling.LANCZOS)
            except:
                print('Cannot resize', self.base_names[idx])
                quit(1)
        if self.charset is not None:
            text = torch.Tensor([self.charset[x] for x in open('%s.txt' % self.base_names[idx]).read().strip()])
        else:
            text = open('%s.txt' % self.base_names[idx]).read().strip()
        
        if self.cache is not None:
            self.cache[idx] = (im, text)
        if self.transform:
            im = self.transform(im)
        return im, text


class OFACRDataset(Dataset):
    def __init__(self, folder, charset, transform=None, height=32, cache=True):
        self.base_names = [os.path.join(folder, x.replace('.txt', '')) for x in os.listdir(folder) if x.endswith('.txt')]
        self.transform = transform
        self.height = height
        self.charset = charset
        self.cache = {} if cache else None

    def __len__(self):
        return len(self.base_names)

    def __getitem__(self, idx):
        if self.cache is not None and idx in self.cache:
            im, text = self.cache[idx]
            if self.transform:
                im = self.transform(im)
            return im, text
        im = Image.open('%s.jpg' % self.base_names[idx]).convert('RGB')
        if im.size[1]!=self.height:
            ratio = self.height / im.size[1]
            width = int(im.size[0] * ratio)
            try:
                im = im.resize((width,self.height), Image.Resampling.LANCZOS)
            except:
                print('Cannot resize', self.base_names[idx])
                quit(1)
        if self.charset is not None:
            text = torch.Tensor([self.charset[x] for x in open('%s.txt' % self.base_names[idx]).read().strip()])
        else:
            text = open('%s.txt' % self.base_names[idx]).read().strip()
        
        if self.cache is not None:
            self.cache[idx] = (im, text)
        if self.transform:
            im = self.transform(im)
        return im, text

if __name__=='__main__':
    cs = json.load(open('../../bounding_boxes/chars.json', 'rt'))
    ds = OCRDataset(folder='lines', charset=cs)
    a, b = ds.__getitem__(0)
    print(a)
    print(b)
