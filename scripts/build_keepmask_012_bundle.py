#!/usr/bin/env python3
"""Build the paper keep-mask {0,11,12} BTH group bundle."""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
for split in ('train','val','test'):
 rows=[json.loads(x) for x in (a.source/'assets'/split/'group_index.jsonl').read_text().splitlines() if x]
 outidx=a.output/'assets'/split/'group_index.jsonl';outidx.parent.mkdir(parents=True,exist_ok=True)
 with outidx.open('w') as f:
  for row in rows:
   group=outidx.parent/row['group_id'];group.mkdir(exist_ok=True)
   raw=np.load(row['raw_ts_path']).astype(np.float32)[...,[0,11,12]]
   rawpath=group/'raw_ts_keep012.npy';np.save(rawpath,raw)
   row['raw_ts_path']=str(rawpath)
   print(json.dumps(row),file=f)
 print(split,len(rows))
