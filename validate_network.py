import sys
import json
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import editdistance

# ~ from ocr_dataset import OCRDataset
from orfac_dataset import ORFACDataset
# ~ from ocr_dataset import collate_batch
from orfac_dataset import orfac_collate_batch
from network import OCROnly

from converter import Converter

def validate_loaded_net(socr, data_path):
    device = 'cuda:0'
    cs        = json.load(open('charmap.json', 'rt'))
    cs['/PAD/'] = len(cs)+1

    trans = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])
    dataset = ORFACDataset(folder=data_path, charset=None, transform=trans)
    
    converter = Converter(cs)
    test_dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=orfac_collate_batch)
    
    socr = socr.to(device)
    with torch.no_grad():
        socr.eval()
        d_sum = 0
        c_sum = 0
        for tns, base_width, lbl, _, cf, pf in test_dataloader:
            tns = tns.to(device)
            out = socr(tns)
            out = out.transpose(0,1)
            am = torch.argmax(out[:, :, :], 2)
            res = converter.decode(am, base_width)
            for i in range(len(lbl)):
                ed = editdistance.eval(res[i], lbl[i])
                ll = len(lbl[i])
                d_sum += ed
                c_sum += ll
    if c_sum==0:
        print()
        print()
        print('Error', data_path)
        print()
        print()
        print()
        quit()
    return 100*d_sum/c_sum

def validate_network(model_path, data_path, display=False):
    device = 'cuda:0'
    batch = 32

    cs          = json.load(open('charmap.json', 'rt'))
    cs['/PAD/'] = len(cs)+1

    trans = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])
    dataset = ORFACDataset(folder=data_path, charset=None, transform=trans)
    
    if len(dataset)==0:
        raise Exception()

    network = OCROnly(nb_classes=(len(cs)+1), feature_dim=128).to(device)
    try:
        # ~ network.load_state_dict(torch.load(model_path, map_location=device))
        network.load(model_path)
    except:
        print('Cannot load network')
        quit(1)

    converter = Converter(cs)
    test_dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=orfac_collate_batch)

    with torch.no_grad():
        network.eval()
        d_sum = 0
        c_sum = 0
        for tns, base_width, lbl, _, cf, pf in test_dataloader:
            tns = tns.to(device)
            out = network(tns)
            out = out.transpose(0,1)
            am = torch.argmax(out[:, :, :], 2)
            res = converter.decode(am, base_width)
            for i in range(len(lbl)):
                ed = editdistance.eval(res[i], lbl[i])
                ll = len(lbl[i])
                d_sum += ed
                c_sum += ll
                if display:
                    print()
                    print(lbl[i])
                    print(res[i])
                    print('%d/%d=%.1f%%' % (ed, ll, 100*ed/ll))
    if display:
        print('CER: %.2f' % (100*d_sum/c_sum))
    return 100*d_sum/c_sum

if __name__=='__main__':
    model_path = sys.argv[1]
    font_group = sys.argv[2]
    validate_network(model_path, 'data/valid/single/%s' % font_group, display=True)
