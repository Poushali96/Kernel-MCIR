# Colab cells: Kernel-MCIR rebuttal runs (v3)

Make a new Colab notebook and paste each block below into its own cell, in order.
Before cell 3, upload `kmcir_rebuttal_runs.py` (saved in this project folder) into `MyDrive/kmcir_rebuttal/`, or paste it into a cell that starts with `%%writefile kmcir_rebuttal_runs.py`. The `.ipynb` version was sent in the chat.

# Kernel-MCIR rebuttal runs — v2
Run top to bottom. Results are saved to Drive after every seed.

**v2 change:** SAGE was making the run take ~75 h. The main run (cell 5) now runs **without SAGE** (~10 min/seed). SAGE has its own short run (cell 6): only k=1, p=0, retrained model, 3 seeds, 200 eval points.

```python
# 1) Mount Google Drive and create the working folder
from google.colab import drive
drive.mount('/content/drive')
import os
WORK = '/content/drive/MyDrive/kmcir_rebuttal'
os.makedirs(f'{WORK}/data', exist_ok=True)
%cd {WORK}
DATA = f'{WORK}/data'
print('working in', os.getcwd())
```

```python
# 2) Optional SAGE baseline
!pip -q install sage-importance
```

### (script cell)
```
%%writefile kmcir_rebuttal_runs.py
# paste the full contents of kmcir_rebuttal_runs.py here
```

```python
# 3) Self-test (~1-2 min). Expect s_1 = 0.750
!python kmcir_rebuttal_runs.py selftest 2>&1 | grep -v -i warn | tail -12
```

```python
# 4) Data (cached on Drive; skips download if already there)
from sklearn.datasets import fetch_20newsgroups
for subset in ('train', 'test'):
    print('20NG', subset, len(fetch_20newsgroups(subset=subset, remove=('headers','footers','quotes'), data_home=DATA).data))
import zipfile, urllib.request
aq = f'{DATA}/AirQualityUCI.csv'
if not os.path.exists(aq):
    zp = f'{DATA}/air_quality.zip'
    urllib.request.urlretrieve('https://archive.ics.uci.edu/static/public/360/air+quality.zip', zp)
    zipfile.ZipFile(zp).extractall(DATA)
print('AirQuality exists:', os.path.exists(aq))
```

## P0 · Main dilution run (no SAGE) — ~10 min per seed
**Auto-resume:** if Colab disconnects, re-run cell 1 and the script cell, then re-run this same cell. Finished seeds are skipped. Use `--fresh` only to start over.

```python
# 5) Main run — re-run this same cell after a disconnect; finished seeds are skipped
!python kmcir_rebuttal_runs.py dilution --seeds 10 --B 200 \
    --data-home {DATA} --out dil_main.json 2>&1 | grep -v -i warn
```

## P1 · SAGE baseline only where it is needed (k=1, p=0, retrained) — 3 seeds

```python
# 6) SAGE run (each SAGE call ~5-15 min with 200 points; 2 calls per seed)
!python kmcir_rebuttal_runs.py dilution --seeds 3 --B 200 --k 1 --p 0 --settings retrain --sage \
    --sage-n 200 --sage-thresh 0.1 --data-home {DATA} --out dil_sage.json 2>&1 | grep -v -i warn
```

## P1 · Screening sensitivity (reviewer)

```python
# 7) DF-200 and LR-500, k=1, p=0
!python kmcir_rebuttal_runs.py dilution --seeds 10 --B 200 --screen df200 --k 1 --p 0 --data-home {DATA} --out dil_df200.json 2>&1 | grep -v -i warn
!python kmcir_rebuttal_runs.py dilution --seeds 10 --B 200 --screen lr500 --k 1 --p 0 --data-home {DATA} --out dil_lr500.json 2>&1 | grep -v -i warn
```

## P0 · Table 8 re-run (corrected RFF)

```python
# 8) Synthetic + 20NG + UCI Air Quality
!python kmcir_rebuttal_runs.py rff --datasets synthetic 20ng --airquality {DATA}/AirQualityUCI.csv \
    --D 64 128 256 512 1024 --B 200 --data-home {DATA} --out table8_rff.json 2>&1 | grep -v -i warn
```

## Report

```python
# 9) Tables for the [FILL] slots
!python kmcir_rebuttal_runs.py report dil_main.json --label main_lr200
!python kmcir_rebuttal_runs.py report dil_sage.json --label sage
!python kmcir_rebuttal_runs.py report dil_df200.json --label df200
!python kmcir_rebuttal_runs.py report dil_lr500.json --label lr500
import json, pandas as pd
print(pd.DataFrame(json.load(open('table8_rff.json'))).round(3).to_string(index=False))
```

```python
# 10) Compact markdown table for OpenReview (k=1, p=0, retrained model)
import json, pandas as pd, os
main = pd.DataFrame(json.load(open('dil_main.json')))
parts = [main[main.method != 'SAGE']]
if os.path.exists('dil_sage.json'):
    sg = pd.DataFrame(json.load(open('dil_sage.json')))
    parts.append(sg[sg.method == 'SAGE'])          # SAGE rows come only from the SAGE run
rows = pd.concat(parts)
sel = rows[(rows.k==1)&(rows.p==0)&(rows.setting=='retrain')&(~rows.method.str.startswith('_'))]
g = sel.groupby('method').agg(dil_m=('dilution','mean'), dil_s=('dilution','std'),
                              auc_m=('auroc','mean'), auc_s=('auroc','std'), n=('seed','nunique'))
print('| method | seeds | dilution post/pre | AUROC dup-detect |\n|---|---|---|---|')
for m, r in g.iterrows():
    print(f'| {m} | {r.n} | {r.dil_m:.3f}±{r.dil_s:.3f} | {r.auc_m:.3f}±{r.auc_s:.3f} |')
```
