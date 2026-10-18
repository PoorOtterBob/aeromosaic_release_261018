#!/usr/bin/env python3
"""Build BTH NoPrecip protocol while preserving oracle parent context."""
import argparse,json,re
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
pattern=re.compile(r'Recent precipitation:.*?Upstream parent-label context:')
for split in ('train','val','test'):
 rows=[json.loads(x) for x in (a.source/'assets'/split/'group_index.jsonl').read_text().splitlines() if x]
 outidx=a.output/'assets'/split/'group_index.jsonl';outidx.parent.mkdir(parents=True,exist_ok=True)
 with outidx.open('w') as f:
  for row in rows:
   prompts=json.loads(Path(row['prompts_path']).read_text())
   revised=[]
   for entry in prompts:
    text=entry['prompt'] if isinstance(entry,dict) else str(entry)
    text,n=pattern.subn('Recent precipitation: unavailable. Upstream parent-label context:',text)
    if n != 1: raise ValueError(f'{split}/{row["group_id"]}: invalid replacement')
    revised.append({**entry,'prompt':text} if isinstance(entry,dict) else text)
   d=outidx.parent/row['group_id'];d.mkdir(exist_ok=True);pp=d/'prompts.json';pp.write_text(json.dumps(revised))
   row['prompts_path']=str(pp);print(json.dumps(row), file=f)
 print(split,len(rows))
