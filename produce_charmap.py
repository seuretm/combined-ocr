import os
import json
from tqdm import tqdm

def find_files(directory):
    lst = os.listdir(directory)
    res = []
    for f in lst:
        ff = os.path.join(directory, f)
        if ff.lower().endswith('.txt'):
            res.append(ff)
        elif os.path.isdir(ff):
            res += find_files(ff)
    return res

chars = set()
for txtfile in tqdm(find_files('data')):
    for c in open(txtfile, 'rt'):
        for d in c.strip():
            chars.add(d)

charmap = {}
for c in sorted(chars):
    charmap[c] = len(charmap)+1

json.dump(charmap, open('charmap.json', 'wt'), indent=4)

print('"', end='')
for c in sorted(chars):
    print(c, end='')
print('"')
