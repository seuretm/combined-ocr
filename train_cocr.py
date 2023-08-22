import os
import pickle
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
from network import OCROnly
from network import COCR
from converter import Converter

device = 'cuda:0'
batch = 32
patience = 30

model_path = os.path.join('models', 'cocr')

cs = json.load(open('charmap.json', 'rt'))
cs['/PAD/'] = len(cs)+1
converter = Converter(cs)


augs =[transforms.Grayscale()]
augs.append(transforms.ColorJitter(brightness=0.2, contrast=0.2))
augs.append(transforms.RandomAffine(degrees=0, shear=5))
augs.append(transforms.ToTensor())

trans = transforms.Compose(augs)
ds = ORFACDataset(folder=('data-aug/train/multiple', 'data-aug/train/single/antiqua', 'data-aug/train/single/bastarda', 'data-aug/train/single/fraktur', 'data-aug/train/single/gotico-antiqua', 'data-aug/train/single/greek', 'data-aug/train/single/hebrew', 'data-aug/train/single/italic', 'data-aug/train/single/rotunda', 'data-aug/train/single/schwabacher', 'data-aug/train/single/textura'), charset=cs, transform=trans)
train_dataloader = DataLoader(ds, batch_size=batch, shuffle=True, collate_fn=orfac_collate_batch, num_workers=7)


trans = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor()
])
ds2 = ORFACDataset(folder=('data-aug/valid/multiple', 'data-aug/valid/single/antiqua', 'data-aug/valid/single/bastarda', 'data-aug/valid/single/fraktur', 'data-aug/valid/single/gotico-antiqua', 'data-aug/valid/single/italic', 'data-aug/valid/single/rotunda', 'data-aug/valid/single/schwabacher', 'data-aug/valid/single/textura'), charset=None, transform=trans)
valid_dataloader = DataLoader(ds2, batch_size=1, shuffle=False, collate_fn=orfac_collate_batch, num_workers=7)


network = COCR(
    OCROnly(nb_classes=13, feature_dim=32, lstm_layers=1).to(device),
    {n: OCROnly(nb_classes=(len(cs)+1), feature_dim=128).to(device) for n in range(13)}
).to(device)
try:
    network.load(model_path)
    print('Previous weights reloaded')
except:
    print('Could not load previous model, loading default ones')
    network.classifier.load(os.path.join('models', 'sequence_classifier'))
    for n in tqdm(sorted(network.models), desc='Loading OCR models'):
        network.models[n].load(os.path.join('models', 'all_heavy_aug'))
    print('Default weights loaded')

ctc_loss = torch.nn.CTCLoss(zero_infinity=True)
# ~ optimizer = torch.optim.SGD(network.parameters(), lr=0.0001, momentum=0.9)
# ~ params = [p for p in network.classifier.parameters()]
# ~ for n in network.models:
    # ~ params += [p for p in network.models[n].parameters()]
# ~ optimizer = torch.optim.Adam(params, lr=0.0001)
optimizers = [torch.optim.Adam(network.models[n].parameters(), lr=0.0001) for n in sorted(network.models)]
for o in optimizers:
    o.load_state_dict(torch.load(os.path.join('models', 'all_heavy_aug', 'optimizer.pth')))
optimizers.append(torch.optim.Adam(network.classifier.parameters(), lr=0.0001))
# ~ optimizers[-1].load_state_dict(torch.load(os.path.join('models', 'all_heavy_aug', 'optimizer.pth')))

schedulers = [torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5) for optimizer in optimizers]

cer=100
os.makedirs(model_path, exist_ok=True)
with open(os.path.join(model_path, 'logs.txt'), 'wt') as lfile:
    # ~ with torch.no_grad():
        # ~ network.eval()
        # ~ d_sum = 0
        # ~ c_sum = 0
        # ~ for tns, base_width, lbl, _, cf, pf in tqdm(valid_dataloader, desc='Validation', leave=False):
            # ~ out = network(tns.to(device)).transpose(0,1)
            # ~ am = torch.argmax(out[:, :, :], 2)
            # ~ res = converter.decode(am, base_width)
            # ~ for i in range(len(lbl)):
                # ~ d_sum += editdistance.eval(res[i], lbl[i])
                # ~ c_sum += len(lbl[i])
        # ~ best_cer = (100*d_sum/c_sum)
    # ~ print('Initial CER:', best_cer)
    best_cer = 100
    
    no_imp = 0
    for epoch in range(1000):
        loss_sum = 0
        network.train()
        batches = 0
        for tns, base_width, lbl, base_length, cf, pf in tqdm(train_dataloader, desc='COCR'):
            tns = tns.to(device)
            lbl = lbl.to(device)
            out = network(tns)
            il = network.convert_widths(base_width, out.shape[0])
            ol = torch.Tensor([l for l in base_length]).long()
            loss = ctc_loss(out.log_softmax(2), lbl, input_lengths=il, target_lengths=ol)
            for optimizer in optimizers:
                optimizer.zero_grad()
            loss.backward()
            for optimizer in optimizers:
                optimizer.step()
            loss_sum += loss.item()
            batches += 1
        
        with torch.no_grad():
            network.eval()
            d_sum = 0
            c_sum = 0
            for tns, base_width, lbl, _, cf, pf in tqdm(valid_dataloader, desc='Validation', leave=False):
            # ~ for tns, base_width, lbl, _, cf, pf in valid_dataloader:
                out = network(tns.to(device)).transpose(0,1)
                am = torch.argmax(out[:, :, :], 2)
                res = converter.decode(am, base_width)
                for i in range(len(lbl)):
                    d_sum += editdistance.eval(res[i], lbl[i])
                    c_sum += len(lbl[i])
            cer = (100*d_sum/c_sum)
            for scheduler in schedulers:
                scheduler.step(cer)
            tqdm.write('Loss sum: %.6f' % (loss_sum/batches))
            tqdm.write('     CER: %.2f' % cer)
            lfile.write('%d;%f;%f\n' % (epoch, loss_sum, cer))
            lfile.flush()
            if cer<=best_cer:
                no_imp = 0
                network.save(model_path)
                best_cer = cer
            else:
                no_imp += 1
            if no_imp>patience:
                print('No improvement, lowest CER: %.2f' % best_cer)
                break
