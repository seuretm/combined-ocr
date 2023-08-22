import os
import json
import sys
import editdistance
import json
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import tormentor
from tormentor.random import Uniform

from orfac_dataset import ORFACDataset
from orfac_dataset import orfac_collate_batch
from network import OCROnly
from converter import Converter

tormentor.Wrap.intensity = Uniform(value_range=(0., 1.5))

config = json.load(open(sys.argv[1], 'rt'))
for arg in sys.argv[2:]:
    spl = arg.split('=')
    config[spl[0]] = spl[1]

device = 'cuda:0'
batch = 32
patience = 20

cs = json.load(open('charmap.json', 'rt'))
cs['/PAD/'] = len(cs)+1
converter = Converter(cs)


augs =[transforms.Grayscale()]
for a in config['augmentations']:
    if   a[0]=='ColorJitter':
        augs.append(transforms.ColorJitter(brightness=a[1], contrast=a[2]))
    elif a[0]=='Affine':
        augs.append(transforms.RandomAffine(degrees=a[1], shear=a[2]))
    else: raise Exception('Unknown transform: %s' % a[0])
augs.append(transforms.ToTensor())
#augs.append(tormentor.Wrap())

trans = transforms.Compose(augs)
ds = ORFACDataset(folder=config['training'], charset=cs, transform=trans)
# ~ ds = OCRDataset(folder='/dev/shm/antiqua', charset=cs, transform=trans)
train_dataloader = DataLoader(ds, batch_size=batch, shuffle=True, collate_fn=orfac_collate_batch, num_workers=7)


trans = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor()
])
ds2 = ORFACDataset(folder=config['validation'], charset=None, transform=trans)
valid_dataloader = DataLoader(ds2, batch_size=1, shuffle=False, collate_fn=orfac_collate_batch, num_workers=7)


network = OCROnly(nb_classes=(len(cs)+1), feature_dim=128).to(device)
print(network)
if config['base'] is not None:
    try:
        network.load(config['base'])
        print('Weights loaded from', config['base'])
    except:
        print('Could not load base:', config['base'])
        quit(1)
else:
    print('Training untrained network')

ctc_loss = torch.nn.CTCLoss(zero_infinity=True)
# ~ optimizer = torch.optim.SGD(network.parameters(), lr=0.0001, momentum=0.9)
optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
# ~ optimizer = torch.optim.SGD(network.parameters(), lr=0.001)

try:
    optimizer.load_state_dict(torch.load(os.path.join(config['base'], 'optimizer.pth')))
except:
    print('No state dict for the optimizer')
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)

os.makedirs(config['filename'], exist_ok=True)
network.save(config['filename'])
torch.save(optimizer.state_dict(), os.path.join(config['filename'], 'optimizer.pth'))


cer=100
with open(os.path.join(config['filename'], 'logs.txt'), 'wt') as lfile:
    # ~ with torch.no_grad():
        # ~ network.eval()
        # ~ d_sum = 0
        # ~ c_sum = 0
        # ~ for tns, base_width, lbl, _ in tqdm(valid_dataloader, desc='Initial validation', leave=False):
            # ~ out = network(tns.to(device)).transpose(0,1)
            # ~ am = torch.argmax(out[:, :, :], 2)
            # ~ res = converter.decode(am, base_width)
            # ~ for i in range(len(lbl)):
                # ~ d_sum += editdistance.eval(res[i], lbl[i])
                # ~ c_sum += len(lbl[i])
        # ~ best_cer = (100*d_sum/c_sum)
        # ~ print('Initial CER: %.2f' % best_cer)
    best_cer = 100
    no_imp = 0
    for epoch in range(1000):
        # ~ if batch==32 and cer<50:
            # ~ batch=320
            # ~ train_dataloader = DataLoader(ds, batch_size=batch, shuffle=True, collate_fn=collate_batch, num_workers=7)
        
        loss_sum = 0
        network.train()
        batches = 0
        for tns, base_width, lbl, base_length, _, _ in tqdm(train_dataloader, desc='%s, %d' % (config['filename'], no_imp)):
            tns = tns.to(device)
            lbl = lbl.to(device)
            out = network(tns)
            il = network.convert_widths(base_width, out.shape[0])
            ol = torch.Tensor([l for l in base_length]).long()
            loss = ctc_loss(out.log_softmax(2), lbl, input_lengths=il, target_lengths=ol)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            batches += 1
        
        with torch.no_grad():
            network.eval()
            d_sum = 0
            c_sum = 0
            for tns, base_width, lbl, _, _, _ in tqdm(valid_dataloader, leave=False, desc='Validation'):
                out = network(tns.to(device)).transpose(0,1)
                am = torch.argmax(out[:, :, :], 2)
                res = converter.decode(am, base_width)
                for i in range(len(lbl)):
                    d_sum += editdistance.eval(res[i], lbl[i])
                    c_sum += len(lbl[i])
            cer = (100*d_sum/c_sum)
        scheduler.step(cer)
        tqdm.write('Loss sum: %.6f' % (loss_sum/batches))
        tqdm.write('     CER: %.2f' % cer)
        lfile.write('%d;%f;%f\n' % (epoch, loss_sum, cer))
        lfile.flush()
        if cer<best_cer:
            no_imp = 0
            network.save(config['filename'])
            torch.save(optimizer.state_dict(), os.path.join(config['filename'], 'optimizer.pth'))
            best_cer = cer
            print('Saved')
        else:
            no_imp += 1
        if no_imp>patience:
            print('No improvement, lowest CER: %.2f' % best_cer)
            break
