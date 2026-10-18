#!/usr/bin/env python3
"""LEGACY AUDIT ONLY: rebuild the preserved weighted BTH historical bundle.

This reconstructs an older asset for provenance comparison. Its weighted DAG,
short precipitation prompt and NPZ inputs are not the paper-aligned release
protocol. Use prepare_camera_ready_data.py for new camera-ready bundles.
"""
import argparse, json
from collections import deque
from pathlib import Path
import numpy as np


def coords(path):
    out = {}
    for line in Path(path).read_text().splitlines():
        _, city, lon, lat = line.split()
        out[city] = (float(lon), float(lat))
    return out


def dag(x, xy):
    # Destination-by-source adjacency. Last historical u/v wind is channels 7/8.
    n, a = len(x), np.zeros((len(x), len(x)), np.float32)
    for dst in range(n):
        ranked = []
        for src in range(n):
            if src == dst: continue
            lon0, lat0 = xy[src]; lon1, lat1 = xy[dst]
            east = (lon1-lon0)*111*np.cos(np.deg2rad((lat0+lat1)/2)); north = (lat1-lat0)*111
            distance = float(np.hypot(east, north)); wind = x[src, -1, [7,8]].astype(float)
            speed = float(np.linalg.norm(wind))
            if 0 < distance <= 600 and speed > 1e-8:
                align = float(np.dot(wind, [east,north])/(speed*distance))
                if align >= .25: ranked.append((align*speed/max(distance,1), src))
        for score, src in sorted(ranked, reverse=True)[:3]: a[dst,src] = score
    # Delete weakest edge until source->destination graph is acyclic.
    while True:
        state = np.zeros(n, np.int8); stack=[]; found=[]
        def visit(src):
            state[src]=1; stack.append(src)
            for dst in np.flatnonzero(a[:,src]):
                if state[dst] == 0: visit(int(dst))
                elif state[dst] == 1 and not found: found.extend(stack[stack.index(int(dst)):]+[int(dst)])
                if found: return
            stack.pop(); state[src]=2
        for i in range(n):
            if state[i] == 0: visit(i)
            if found: break
        if not found: break
        edges=[(found[k+1],found[k]) for k in range(len(found)-1)]
        d,s=min(edges,key=lambda e:a[e]); a[d,s]=0
    indeg=(a>0).sum(1).astype(np.int64); remain=indeg.copy(); level=np.zeros(n,np.int64); q=deque(np.flatnonzero(indeg==0))
    while q:
        src=q.popleft()
        for dst in np.flatnonzero(a[:,src]):
            level[dst]=max(level[dst],level[src]+1); remain[dst]-=1
            if remain[dst]==0:q.append(int(dst))
    return a, level, (indeg==0).astype(np.int64)


def precip(history):
    v=float(history[:,6].mean())
    return 'low' if v < -.5 else ('moderate' if v < .5 else 'high')


def main():
    p=argparse.ArgumentParser(); p.add_argument('--source-dir',type=Path,required=True); p.add_argument('--output-dir',type=Path,required=True); p.add_argument('--city-file',type=Path,required=True); p.add_argument('--max-groups',type=int,default=0); p.add_argument('--allow-legacy-protocol',action='store_true'); args=p.parse_args()
    if not args.allow_legacy_protocol:
        p.error('legacy weighted BTH reconstruction only; pass --allow-legacy-protocol to acknowledge it is not the paper protocol')
    all_xy=coords(args.city_file)
    for split in ('train','val','test'):
        z=np.load(args.source_dir/'npz'/f'{split}.npz',allow_pickle=True); dates=z['target_date'].astype(str); unique=list(dict.fromkeys(dates.tolist()))
        if args.max_groups: unique=unique[:args.max_groups]
        index=args.output_dir/'assets'/split/'group_index.jsonl'; index.parent.mkdir(parents=True,exist_ok=True)
        with index.open('w') as fp:
            for date in unique:
                mask=dates==date; x=z['timeseries'][mask].astype(np.float32); y=z['labels'][mask].astype(np.int64); cities=z['city_name'][mask].astype(str).tolist(); xy=[all_xy[c] for c in cities]
                a,l,r=dag(x,xy); out=index.parent/date; out.mkdir(exist_ok=True)
                files={'raw_ts_path':'raw_ts.npy','labels_path':'labels.npy','dag_adj_path':'dag_adj.npy','dag_level_path':'dag_level.npy','is_root_path':'is_root.npy','parent_count_path':'parent_count.npy','city_names_path':'city_names.json','prompts_path':'prompts.json'}
                np.save(out/files['raw_ts_path'],x); np.save(out/files['labels_path'],y); np.save(out/files['dag_adj_path'],a); np.save(out/files['dag_level_path'],l); np.save(out/files['is_root_path'],r); np.save(out/files['parent_count_path'],(a>0).sum(axis=1).astype(np.int64))
                (out/files['city_names_path']).write_text(json.dumps(cities)); prompts=[]
                for dst,city in enumerate(cities):
                    parents=np.flatnonzero(a[dst]).tolist(); context='no upstream parent' if not parents else ', '.join(f'{cities[i]}: class {int(y[i])}' for i in parents)
                    prompts.append({'city_name':city,'prompt':f'Region: BTH. City: {city}. Target date: {date}. DAG level: {int(l[dst])}. Recent precipitation: {precip(x[dst])}. Upstream parent-label context: {context}. Predict the PM2.5 risk class.'})
                (out/files['prompts_path']).write_text(json.dumps(prompts)); row={'group_id':date,'target_date':date,**{k:str(out/v) for k,v in files.items()}}; fp.write(json.dumps(row)+'\n')
        print(f'{split}: {len(unique)} groups')
if __name__ == '__main__': main()
