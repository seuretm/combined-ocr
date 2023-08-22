import os
import json
import sys
import editdistance
import json
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from orfac_dataset import ORFACDataset
from orfac_dataset import orfac_collate_batch
from network import NoDimRedBackbone
from network import ColClassifier


device = 'cuda:0'
batch = 1
patience = 20

model_path = os.path.join('models', 'classifier')

cs = json.load(open('charmap.json', 'rt'))
cs['PAD'] = len(cs)+1
#converter = Converter(cs)


augs =[transforms.Grayscale(), transforms.ColorJitter(brightness=0.2, contrast=0.2)]
augs.append(transforms.ToTensor())

# trained on multiple only, batch 1, until 90%, then all, batch 16
trans = transforms.Compose(augs)
ds = ORFACDataset(folder=('data-aug/train/multiple', 'data-aug/train/single/antiqua', 'data-aug/train/single/bastarda', 'data-aug/train/single/fraktur', 'data-aug/train/single/gotico-antiqua', 'data-aug/train/single/greek', 'data-aug/train/single/hebrew', 'data-aug/train/single/italic', 'data-aug/train/single/rotunda', 'data-aug/train/single/schwabacher', 'data-aug/train/single/textura'), charset=cs, transform=trans)
# ~ ds = ORFACDataset(folder=('data/train/multiple', ), charset=cs, transform=trans)
train_dataloader = DataLoader(ds, batch_size=batch, shuffle=True, collate_fn=orfac_collate_batch, num_workers=7)

ds = ORFACDataset(folder=('data/valid/multiple', 'data/valid/single/antiqua', 'data/valid/single/bastarda', 'data/valid/single/fraktur', 'data/valid/single/gotico-antiqua', 'data/valid/single/italic', 'data/valid/single/rotunda', 'data/valid/single/schwabacher', 'data/valid/single/textura'), charset=cs, transform=trans)
valid_dataloader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=orfac_collate_batch, num_workers=7)

network = ColClassifier(NoDimRedBackbone(), feature_dim=24, nb_classes=13).to(device)
try:
    network.load(model_path)
    print('Previous weights reloaded')
except: pass


loss_w = torch.Tensor([1.0 if x>0 else 0.0 for x in range(13)]).to(device)
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)
# ~ optimizer = torch.optim.SGD(network.parameters(), lr=0.1, momentum=0.9)
# trained with adam 0.0001 until 97.4, then highest lr
optimizer = torch.optim.Adam(network.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)

cer=100
os.makedirs(model_path, exist_ok=True)
with open(os.path.join(model_path, 'logs.txt'), 'wt') as lfile:
    network.eval()    
    good = 0
    bad  = 0
    for tns, base_im_width, txt, base_txt_length, cf, pf in tqdm(valid_dataloader, desc='Initial validation', leave=False):
        out = network(tns.to(device))
        m, idx = torch.max(out, 2)
        idx = idx.permute(1,0).to(device)
        pf = pf.to(device)
        good += (idx==pf).sum().item()
        bad  += (idx!=pf).sum().item()
    best_acc = (100*good/(good+bad))
    print('Initial accuracy: %.2f' % best_acc)
    no_imp = 0
    for epoch in range(1000):
        
        loss_sum = 0
        network.train()
        batches = 0
        good = 0
        bad  = 0
        count = [0 for x in range(12)]
        network.train()
        for tns, base_im_width, txt, base_txt_length, cf, pf in tqdm(train_dataloader, desc='Epoch %d' % epoch, leave=True):
            out = network(tns.to(device))
            m, idx = torch.max(out, 2)
            loss = loss_fn(out.reshape(-1,13), pf.view(-1).to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            idx = idx.permute(1,0).to(device)
            pf = pf.to(device)
            good += (idx==pf).sum().item()
            bad  += (idx!=pf).sum().item()
            for x in range(12):
                count[x] += (idx==x).sum().item()
        acc = (100*good/(good+bad))
        
        
        network.eval()
        for tns, base_im_width, txt, base_txt_length, cf, pf in tqdm(valid_dataloader, desc='Validation', leave=False):
            out = network(tns.to(device))
            m, idx = torch.max(out, 2)
            loss = loss_fn(out.reshape(-1,13), pf.view(-1).to(device))
            idx = idx.permute(1,0).to(device)
            pf = pf.to(device)
            good += (idx==pf).sum().item()
            bad  += (idx!=pf).sum().item()
            for x in range(12):
                count[x] += (idx==x).sum().item()
        acc = (100*good/(good+bad))
        scheduler.step(100-acc)
        
        if best_acc is None or acc>best_acc:
            best_acc = acc
            network.save(model_path)
            torch.save(optimizer.state_dict(), os.path.join(model_path, 'optimizer.pth'))
            print('Saved with acc %f' % best_acc)
        else:
            print('Not saved')
        print('loss %f' % loss_sum)
        print('Good: %6d' % good)
        print(' Bad: %6d' % bad)
        print(' Acc: %.2f' % acc)
        print(count)
