#!/usr/bin/env python3
"""Create six paper-protocol bundles from normalized history and raw precipitation.

The normalized 13-channel history cannot recover precipitation in metres. A
separate raw precipitation array is therefore required for the paper's text
construction rule. This script does not convert upstream KnowAir.npy to his.npz.
"""
import argparse, json, math
from collections import deque
from datetime import date, timedelta
from pathlib import Path
import numpy as np

REGIONS = {
"BTH":["Beijing","Tianjin","Shijiazhuang","Tangshan","Qinhuangdao","Handan","Baoding","Zhangjiakou","Chengde","Langfang","Cangzhou","Hengshui","Xingtai"],
"YRD":["Shanghai","Nanjing","Suzhou","Nantong","Yangzhou","Wuxi","Changzhou","Zhenjiang","Hangzhou","Ningbo","Shaoxing","Huzhou","Jiaxing","Hefei","Wuhu","Maanshan"],
"SDP":["Jinan","Qingdao","Laiwu","Linyi","Liaocheng","Binzhou","Zibo","Zaozhuang","Yantai","Weifang","Jining","Taian","Rizhao","Dongying","Heze"],
"ZHD":["Zhengzhou","Luoyang","Anyang","Kaifeng","Jiaozuo","Pingdingshan","Sanmenxia","Xinyang","Zhoukou","Hebi","Xinxiang","Puyang","Xuchang","Luohe","Nanyang","Shangqiu","Zhumadian"],
"CJD":["Nanchang","Wuhan","Jiujiang","Yichang","Jingzhou","Yueyang","Changde","Zhangjiajie","Jingdezhen","Yingtan","Shangrao","Huangshi","Shiyan","Xiangfan","Ezhou","Jingmen","Xiaogan","Huanggang","Xianning","Suizhou","Yiyang"],
"CYB":["Chongqing","Chengdu","Deyang","Mianyang","Suining","Nanchong","Guangan","Ziyang","Neijiang","Zigong","Leshan","Meishan","Luzhou","Yibin","Dazhou","Guangyuan","Yaan"]}
ALIASES={"Chongqing":"Zhongqing","Chengdu":"Chengdou"}
BINS=np.array([35.,75.,115.,150.,250.])
PRECIP_BINS=(0.1,10.,25.,50.,100.,250.)
PRECIP_NAMES=("no noticeable precipitation","light rain","moderate rain",
              "heavy rain","rainstorm","severe rainstorm","extreme rainstorm")

def precipitation_text(metres):
    """Deterministic eight-step summary of unstandardized precipitation."""
    mm=np.asarray(metres,dtype=np.float64)*1000.
    if mm.shape!=(8,) or not np.all(np.isfinite(mm)) or np.any(mm<0):
        raise ValueError("precipitation must be eight finite nonnegative values in metres")
    total=float(mm.sum())
    if total<PRECIP_BINS[0]:
        return "No noticeable recent precipitation. Dry conditions persisted."
    category=PRECIP_NAMES[int(np.searchsorted(PRECIP_BINS,total,side="right"))]
    segments=(mm[:2],mm[2:5],mm[5:8])
    sums=[float(s.sum()) for s in segments]
    categories=[PRECIP_NAMES[int(np.searchsorted(PRECIP_BINS,v*8/len(s),side="right"))]
                for s,v in zip(segments,sums)]
    parts=[f"Recent {category} was observed in the previous input window.",
           "Early, middle, and late segments: "+", ".join(categories)+"."]
    if sums[0]<sums[1]<sums[2]:
        parts.append("Precipitation increased across the window.")
    elif sums[0]>sums[1]>sums[2]:
        parts.append("Precipitation decreased across the window.")
    else:
        parts.append("The largest raw sum occurred in the "+("early","middle","late")[int(np.argmax(sums))]+" segment.")
    parts.append(f"Rain occurred at {int(np.count_nonzero(mm>=0.1))} of eight observations.")
    return " ".join(parts)

def oracle_prompt(region,city,dt,level,parents,precip_text):
    parent_block=("Upstream: none (root)." if not parents else
                  "Upstream:\n"+"\n".join(f"- {name}: label={label}" for name,label in parents))
    return (f"You are given multimodal air-quality risk-inference features for region {region}.\n"
            "The target city's history is provided through the time-series input, and recent precipitation context is provided separately in text form.\n"
            f"Target city: {city}  Target date: {dt}  Graph level: {level}\n"
            "Use the time-series input as the main evidence for the target city.\n"
            f"Upstream graph context (oracle protocol: target-date parent labels): {parent_block}\n"
            f"Recent precipitation context: {precip_text}\n"
            "Infer the final air-quality risk label for the target city on the target date. Output only one label token: 0, 1, 2, 3, 4, or 5.")

def city_file(path):
    ids,xy={},{}
    for no,line in enumerate(path.read_text(encoding="utf-8").splitlines(),1):
        f=line.split()
        if not f: continue
        if len(f)!=4: raise ValueError(f"{path}:{no}: expected index city lon lat")
        i,n,lon,lat=f; ids[n]=int(i); xy[n]=(float(lon),float(lat))
    return ids,xy

def find_cycle(a):
    n=len(a); state=np.zeros(n,np.int8); stack=[]; pos={}
    def dfs(src):
        state[src]=1; pos[src]=len(stack); stack.append(src)
        for dst in np.flatnonzero(a[:,src]>0):
            dst=int(dst)
            if state[dst]==0:
                c=dfs(dst)
                if c: return c
            elif state[dst]==1:
                nodes=stack[pos[dst]:]+[dst]
                return list(zip(nodes[:-1],nodes[1:]))
        stack.pop(); pos.pop(src,None); state[src]=2
        return None
    for i in range(n):
        if state[i]==0:
            c=dfs(i)
            if c: return c
    return None

def dag(x,xy):
    # x: cities x 8 x 13; destination-by-source adjacency.
    n=len(xy); a=np.zeros((n,n),np.float32)
    for dst,(lon_d,lat_d) in enumerate(xy):
        cand=[]
        for src,(lon_s,lat_s) in enumerate(xy):
            if src==dst: continue
            u,v=x[src,-1,7:9].astype(np.float64); speed=float(np.hypot(u,v))
            if not np.isfinite(speed) or speed<=1e-8: continue
            east=(lon_d-lon_s)*111*math.cos(math.radians((lat_s+lat_d)/2))
            north=(lat_d-lat_s)*111; dist=float(math.hypot(east,north))
            if dist<=0 or dist>600: continue
            align=float((u*east+v*north)/(speed*dist))
            if align>=.25: cand.append((align*speed/max(dist,1.),src))
        for score,src in sorted(cand,reverse=True)[:3]: a[dst,src]=score
    while True:
        c=find_cycle(a)
        if not c: break
        src,dst=min(c,key=lambda e:(a[e[1],e[0]],e[0],e[1])); a[dst,src]=0.
    deg=(a>0).sum(1).astype(np.int64); rem=deg.copy(); lev=np.zeros(n,np.int64)
    q=deque(np.flatnonzero(deg==0).tolist())
    while q:
        src=int(q.popleft())
        for dst in np.flatnonzero(a[:,src]>0):
            dst=int(dst); lev[dst]=max(lev[dst],lev[src]+1); rem[dst]-=1
            if rem[dst]==0: q.append(dst)
    # Scores select parents and cycle deletions; the paper's exported A is binary.
    return (a>0).astype(np.float32),lev,(deg==0).astype(np.int64)

def split(d):
    if d.year in (2015,2016): return "train"
    if d.year==2017: return "val"
    if d.year==2018: return "test"
    raise ValueError(f"date outside split: {d}")

def portable(path,root):
    try: return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as e: raise ValueError("--out-root must be inside --project-root") from e

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--his",type=Path,required=True,help="normalized his.npz with data/mean/std")
    p.add_argument("--precipitation-meters",type=Path,required=True,
                   help="raw .npy: (T,184) precipitation in metres or (T,184,18) KnowAir with precipitation at column 12")
    p.add_argument("--city-file",type=Path,required=True)
    p.add_argument("--project-root",type=Path,default=Path.cwd())
    p.add_argument("--out-root",type=Path,required=True)
    p.add_argument("--max-groups",type=int,default=0,help="smoke subset per split; 0=all")
    z=p.parse_args(); root=z.project_root.resolve()
    out=z.out_root if z.out_root.is_absolute() else root/z.out_root
    portable(out,root)
    if z.max_groups<0: p.error("--max-groups must be nonnegative")
    if out.exists() and any(out.iterdir()): p.error(f"refusing non-empty output: {out}")
    d=np.load(z.his,allow_pickle=False)
    if not {"data","mean","std"}.issubset(d.files): raise ValueError("his.npz must contain data, mean, std")
    data=np.asarray(d["data"],np.float32)
    if data.shape!=(11688,184,13): raise ValueError(f"expected (11688,184,13), got {data.shape}")
    mean,std=float(np.asarray(d["mean"]).item()),float(np.asarray(d["std"]).item())
    if not np.isfinite(mean) or not np.isfinite(std) or std<=0: raise ValueError("invalid PM mean/std")
    raw_precip=np.load(z.precipitation_meters,mmap_mode="r",allow_pickle=False)
    if raw_precip.shape==data.shape[:2]:
        precip=raw_precip
    elif raw_precip.shape==data.shape[:2]+(18,):
        precip=raw_precip[:,:,12]
    else:
        raise ValueError(f"raw precipitation has incompatible shape: {raw_precip.shape}")
    ids,allxy=city_file(z.city_file); out.mkdir(parents=True,exist_ok=True)
    origin=date(2015,1,1); ndays=data.shape[0]//8
    for region,cities in REGIONS.items():
        raw=[ALIASES.get(c,c) for c in cities]
        missing=[c for c in raw if c not in ids]
        if missing: raise ValueError(f"{region}: missing cities in city.txt: {missing}")
        ix=[ids[c] for c in raw]; xy=[allxy[c] for c in raw]
        groups={"train":[],"val":[],"test":[]}
        for day in range(1,ndays):
            dt=origin+timedelta(days=day); groups[split(dt)].append(dt)
        if z.max_groups: groups={s:ds[:z.max_groups] for s,ds in groups.items()}
        bundle=out/region
        if bundle.exists() and any(bundle.iterdir()): raise FileExistsError(bundle)
        for part,dates in groups.items():
            sd=bundle/"assets"/part; sd.mkdir(parents=True,exist_ok=True); rows=[]
            for dt in dates:
                di=(dt-origin).days
                h=data[(di-1)*8:di*8,ix,:].astype(np.float32)
                target=data[di*8:(di+1)*8,ix,:]
                if h.shape!=(8,len(cities),13) or target.shape!=h.shape: raise ValueError(f"bad shape {region} {dt}")
                ts=h.transpose(1,0,2)
                labels=np.digitize((target[:,:,0].astype(np.float64)*std+mean).mean(0),BINS,right=True).astype(np.int64)
                a,lev,rootmask=dag(ts,xy)
                parent_count=(a>0).sum(axis=1).astype(np.int64)
                gd=sd/dt.isoformat(); gd.mkdir(parents=True,exist_ok=True)
                for name,val in {"raw_ts":ts,"labels":labels,"dag_adj":a,"dag_level":lev,"is_root":rootmask,"parent_count":parent_count}.items(): np.save(gd/f"{name}.npy",val)
                (gd/"city_names.json").write_text(json.dumps(cities),encoding="utf-8")
                prompt=[]
                for i,city in enumerate(cities):
                    par=np.flatnonzero(a[i]>0).tolist()
                    precip_history=precip[(di-1)*8:di*8,ix[i]]
                    ptxt=precipitation_text(precip_history)
                    parent_pairs=[(cities[j],int(labels[j])) for j in par]
                    prompt.append({"city_name":city,"prompt":oracle_prompt(region,city,dt,int(lev[i]),parent_pairs,ptxt)})
                (gd/"prompts.json").write_text(json.dumps(prompt,ensure_ascii=False),encoding="utf-8")
                keys={"raw_ts_path":"raw_ts.npy","labels_path":"labels.npy","dag_adj_path":"dag_adj.npy","dag_level_path":"dag_level.npy","is_root_path":"is_root.npy","parent_count_path":"parent_count.npy","city_names_path":"city_names.json","prompts_path":"prompts.json"}
                rows.append({"group_id":dt.isoformat(),"target_date":dt.isoformat(),"num_nodes":len(cities),**{k:portable(gd/v,root) for k,v in keys.items()}})
            (sd/"group_index.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows),encoding="utf-8")
            print(f"{region} {part}: {len(rows)} groups")
        manifest={"region":region,"cities":cities,"source_city_indices":ix,"split_years":{"train":[2015,2016],"val":[2017],"test":[2018]},"input_window":"previous calendar day, 8x13 historical channels","precipitation_text":"raw metres, previous eight steps; seven categories with thresholds 0.1/10/25/50/100/250 mm and segment summaries","dag_wind":"last observation in historical window, channels 7-8","label":"mean target-day PM2.5 in original units; bins 35/75/115/150/250","dag":{"max_distance_km":600,"min_cosine_alignment":.25,"max_parents":3,"cycle_break":"remove lowest-score edge in detected cycle","adjacency":"binary destination-by-source"},"terrain_filter":False,"groups_per_split":{s:len(ds) for s,ds in groups.items()}}
        (bundle/"dataset_manifest.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")

if __name__=="__main__": main()
