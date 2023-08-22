import json
from PIL import Image
import os
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import pickle

cs = json.load(open('charmap.json', 'rt'))
cs['PAD'] = len(cs)+1

no_font = 0

def orfac_collate_batch(batch):
    im_list = []
    txt_list = []
    cf_list = []
    pf_list = []
    # ~ print('>>')
    # ~ print(batch)
    for (im, txt, cf, pf) in batch:
        im_list.append(im.permute(2, 0, 1))
        txt_list.append(txt)
        cf_list.append(cf)
        pf_list.append(pf)
    base_im_width = [im.shape[0] for im in im_list]
    base_txt_length = [len(txt) for txt in txt_list]
    im_list = pad_sequence(im_list, batch_first=False, padding_value=0).permute(1, 2, 3, 0)
    cf_list = pad_sequence(cf_list, batch_first=False, padding_value=no_font)
    pf_list = pad_sequence(pf_list, batch_first=False, padding_value=no_font)
    im_list = torch.stack([x for x in im_list])
    try:
        cf_list = torch.stack([x for x in cf_list])
    except:
        print(cf_list)
        quit()
    pf_list = torch.stack([x for x in pf_list])
    if type(txt)!=str:
        txt_list = pad_sequence(txt_list, batch_first=True, padding_value=cs['PAD'])
    return im_list, base_im_width, txt_list, base_txt_length, cf_list, pf_list

class ORFACDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, folder, charset, transform=None, height=32, cache=True):
        # Image
        # Text
        # Character labels
        # Pixel labels
        if type(folder)==str:
            self.base_names = [os.path.join(folder, x.replace('.txt', '')) for x in os.listdir(folder) if x.endswith('.txt')]
        else: # assuming it's a tuple
            self.base_names = []
            for fld in folder:
                self.base_names += [os.path.join(fld, x.replace('.txt', '')) for x in os.listdir(fld) if x.endswith('.txt')]
        self.transform = transform
        self.height = height
        self.charset = charset
        self.cache = {} if cache else None

    def __len__(self):
        return len(self.base_names)

    def __getitem__(self, idx):
        if self.cache is not None and idx in self.cache:
            im, text, cfont, cfont = self.cache[idx]
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
            try:
                text = torch.Tensor([self.charset[x] for x in open('%s.txt' % self.base_names[idx]).read().strip()])
            except:
                print('Failed to read')
                print('%s.txt' % self.base_names[idx])
                quit(1)
        else:
            text = open('%s.txt' % self.base_names[idx]).read().strip()
        # character font
        cf = torch.Tensor(pickle.load(open('%s.cf' % self.base_names[idx], 'rb'))).long()
        # pixel font
        pf = torch.Tensor(pickle.load(open('%s.pf' % self.base_names[idx], 'rb')))
        pf = F.interpolate(pf.view(1,1,-1).float(), size=(im.size[0],), mode='nearest').view(-1).long()
        
        # wrong cf length?
        if cf.shape[0]!=len(text):
            if cf.shape[0]==0:
                cf = F.interpolate(pf.view(1,1,-1).float(), size=(len(text),), mode='nearest').view(-1).long()
            else:
                cf = F.interpolate(cf.view(1,1,-1).float(), size=(len(text),), mode='nearest').view(-1).long()
            
        
        if self.cache is not None:
            self.cache[idx] = (im, text, cf, pf)
        if self.transform:
            im = self.transform(im)
        return im, text, cf, pf

if __name__=='__main__':
    cs = json.load(open('charmap.json', 'rt'))
    ds = ORFACDataset(folder='data/train/single/antiqua', charset=cs)
    i = ds.__getitem__(0)
    print(i)
