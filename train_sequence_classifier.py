import os
import json
import sys
import editdistance
import json
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F

from orfac_dataset import ORFACDataset
from orfac_dataset import orfac_collate_batch
from network import OCROnly


device = 'cuda:0'
batch = 128
patience = 20

model_path = os.path.join('models', 'sequence_classifier')

cs = json.load(open('charmap.json', 'rt'))
cs['PAD'] = len(cs)+1

augs =[transforms.Grayscale(), transforms.ColorJitter(brightness=0.2, contrast=0.2)]
augs.append(transforms.ToTensor())

trans = transforms.Compose(augs)
ds = ORFACDataset(folder=('data-aug/train/multiple', 'data-aug/train/single/antiqua', 'data-aug/train/single/bastarda', 'data-aug/train/single/fraktur', 'data-aug/train/single/gotico-antiqua', 'data-aug/train/single/greek', 'data-aug/train/single/hebrew', 'data-aug/train/single/italic', 'data-aug/train/single/rotunda', 'data-aug/train/single/schwabacher', 'data-aug/train/single/textura'), charset=cs, transform=trans)
# ~ ds = ORFACDataset(folder=('data-aug/train/multiple', 'data-aug/train/single/greek', 'data-aug/train/single/hebrew', 'data-aug/train/single/gotico-antiqua'), charset=cs, transform=trans)
train_dataloader = DataLoader(ds, batch_size=batch, shuffle=True, collate_fn=orfac_collate_batch, num_workers=7)

ds = ORFACDataset(folder=('data-aug/valid/multiple', 'data-aug/valid/single/antiqua', 'data-aug/valid/single/bastarda', 'data-aug/valid/single/fraktur', 'data-aug/valid/single/gotico-antiqua', 'data-aug/valid/single/italic', 'data-aug/valid/single/rotunda', 'data-aug/valid/single/schwabacher', 'data-aug/valid/single/textura'), charset=cs, transform=trans)
valid_dataloader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=orfac_collate_batch, num_workers=7)

network = OCROnly(nb_classes=13, feature_dim=32, lstm_layers=1).to(device)
# ~ try:
    # ~ network.load(model_path)
    # ~ print('Previous weights reloaded')
# ~ except: pass


# ~ loss_w = torch.Tensor([1.0 if x>0 else 0.0 for x in range(13)]).to(device)
# ~ loss_fn = torch.nn.CrossEntropyLoss(weight=loss_w)
loss_fn = torch.nn.CrossEntropyLoss()
# ~ optimizer = torch.optim.SGD(network.parameters(), lr=0.1, momentum=0.9)
optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)

with open('logs.txt', 'wt') as lfile:
    # ~ best_acc = None
    # ~ with torch.no_grad():
        # ~ network.eval()
        # ~ good = 0
        # ~ bad  = 0
        # ~ for tns, base_im_width, txt, base_txt_length, cf, pf in tqdm(valid_dataloader, desc='Valid %d' % 0, leave=False):
            # ~ out = network(tns.to(device))#.permute(0,2,1)
            # ~ pf = F.interpolate(pf.permute((1,0)).unsqueeze(0).float(), size=out.shape[0], mode='linear').long().squeeze(0)
            # ~ out = out.transpose(1,0).reshape(-1, 13)
            # ~ pf  = pf.view(-1).to(device)
            # ~ m, idx = torch.max(out, 1)
            # ~ idx = idx.to(device)
            # ~ good += (idx==pf).sum().item()
            # ~ bad  += (idx!=pf).sum().item()
        # ~ best_acc = (100*good/(good+bad))
    # ~ print('Initial acc:', best_acc)
    best_acc = 0
    
    no_imp = 0
    for epoch in range(200):
        
        loss_sum = 0
        network.train()
        batches = 0
        good = 0
        bad  = 0
        count = [0 for x in range(12)]
        network.train()
        for tns, base_im_width, txt, base_txt_length, cf, pf in tqdm(train_dataloader, desc='Epoch %d, %d' % (epoch, no_imp), leave=True):
            out = network(tns.to(device))#.permute(0,2,1)
            pf = F.interpolate(pf.permute((1,0)).unsqueeze(0).float(), size=out.shape[0], mode='linear').long().squeeze(0)
            out = out.transpose(1,0).reshape(-1, 13)
            pf  = pf.view(-1).to(device)
            loss = loss_fn(out, pf)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            m, idx = torch.max(out, 1)
            idx = idx.to(device)
            good += (idx==pf).sum().item()
            bad  += (idx!=pf).sum().item()
            for x in range(12):
                count[x] += (idx==x).sum().item()
        acc = (100*good/(good+bad))
        
        network.eval()
        good = 0
        bad  = 0
        for tns, base_im_width, txt, base_txt_length, cf, pf in tqdm(valid_dataloader, desc='Valid %d' % epoch, leave=False):
            out = network(tns.to(device))#.permute(0,2,1)
            pf = F.interpolate(pf.permute((1,0)).unsqueeze(0).float(), size=out.shape[0], mode='linear').long().squeeze(0)
            out = out.transpose(1,0).reshape(-1, 13)
            pf  = pf.view(-1).to(device)
            m, idx = torch.max(out, 1)
            idx = idx.to(device)
            good += (idx==pf).sum().item()
            bad  += (idx!=pf).sum().item()
        acc = (100*good/(good+bad))
        scheduler.step(100-acc)
        
        if best_acc is None or acc>best_acc:
            best_acc = acc
            network.save(model_path)
            print('Saved with acc %f' % best_acc)
            no_imp = 0
        else:
            print('loss %f' % loss_sum)
            no_imp += 1
        print('Good: %6d' % good)
        print(' Bad: %6d' % bad)
        print(' Acc: %.1f' % acc)
        print(count)
        if no_imp>20:
            break
