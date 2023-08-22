import warnings
warnings.filterwarnings("ignore")

import pickle
from tqdm import tqdm
import torch
import json
from network import OCROnly
import sys
import os
from validate_network import validate_network
from validate_network import validate_loaded_net
from network import SelectiveOCR
from network import NoDimRedBackbone
from network import ColClassifier
from network import EnsembleOCR
from network import SplitOCR
from network import OCROnly
from network import COCR
from network import COCR2

from bench import Bench

cs = json.load(open('charmap.json', 'rt'))
cs['/PAD/'] = len(cs)+1

subset = 'test'

device = 'cuda:0'
specific_models = {
    0: 'all_heavy_aug',
    1: 'antiqua_ft_heavy_aug',
    2: 'bastarda_ft_heavy_aug',
    3: 'fraktur_ft_heavy_aug',
    4: 'textura_ft_heavy_aug',
    5: 'schwabacher_ft_heavy_aug',
    # ~ 6: 'greek_ft_heavy_aug',
    7: 'italic_ft_heavy_aug',
    # ~ 8: 'hebrew_ft_heavy_aug',
    9: 'gotico-antiqua_ft_heavy_aug',
    # ~ 10: 'manuscript_ft_heavy_aug',
    11: 'rotunda_ft_heavy_aug'
}
ocr_models = {}
for n in tqdm(specific_models, desc='Loading models', leave=False):
    network = OCROnly(nb_classes=(len(cs)+1), feature_dim=128).to(device)
    network.load(os.path.join('models', specific_models[n]))
    ocr_models[n] = network
classifier = ColClassifier(NoDimRedBackbone(), feature_dim=24, nb_classes=13).to(device)
classifier.load(os.path.join('models', 'classifier'))
selocr = SelectiveOCR(classifier, ocr_models)
splocr = SplitOCR(classifier, ocr_models)


font_groups = [x for x in sorted(os.listdir(os.path.join('data', 'test', 'single')))]
data_path = ['data/%s/all' % subset]+['data/%s/single/%s' % (subset, fg) for fg in font_groups]

bench = Bench(['System', 'All'] + font_groups)

if True:
    ens = EnsembleOCR('ensembles', len(cs)+1, device).to(device)
    bench('Ensemble')
    for dp in tqdm(data_path, desc='Ensemble'):
        cer = validate_loaded_net(ens, dp)
        bench(cer)
    print()
    print()
    print(bench.mattermost())


if True:
    network = COCR(
        OCROnly(nb_classes=13, feature_dim=32, lstm_layers=1).to(device),
        {n: OCROnly(nb_classes=(len(cs)+1), feature_dim=128).to(device) for n in range(13)}
    ).to(device)
    network.load(os.path.join('models', 'cocr'))
    bench('COCR')
    for dp in tqdm(data_path, desc='COCR'):
        cer = validate_loaded_net(network, dp)
        bench(cer)
    print()
    print()
    print(bench.mattermost())


sys.stdout.flush()
if True:
    bench('SelOCR')
    for dp in tqdm(data_path, desc='SelOCR'):
        cer = validate_loaded_net(selocr, dp)
        bench(cer)
    print()
    print()
    print(bench.mattermost())


sys.stdout.flush()
if False:
    bench('SplitOCR')
    for dp in tqdm(data_path, desc='SplitOCR'):
        cer = validate_loaded_net(splocr, dp)
        bench(cer)
    print()
    print()
    print(bench.mattermost())
    
for model_name in sorted(os.listdir('models')):
    if model_name in ('classifier', 'classifier-bak', 'sequence_classifier', 'cocr'):
        continue
    bench(model_name)

    for dp in tqdm(data_path, desc=model_name):
        try:
            cer = validate_network(os.path.join('models', model_name), dp)
            bench(cer)
        except:
            bench('-')
    print()
    print()
    print(bench.mattermost())
