import numpy as np, sys
from rmlm_lib import *
res=[]
for seed in range(3):
    rng=np.random.default_rng(seed)
    X,A,C=simulate_rmlm(30,2285,0.1,seed=seed)
    # truth = minimum DAG of A
    run=lambda M: pipeline(M,1.3,0.1,int(.11*len(M)),int(.04*len(M)),0.1)
    e0=run(X)[0]
    # truth edges via hard threshold on true A with delta=0 on minimum dag
    tr=hard_threshold_dag(A,0.0)
    cnt={}
    B=50
    for b in range(B):
        nb=int(np.ceil(len(X)/20)); st=rng.integers(0,len(X)-20+1,nb)
        Xb=np.concatenate([X[s:s+20] for s in st])[:len(X)]
        for e in run(Xb)[0]: cnt[e]=cnt.get(e,0)+1
    f=lambda t:{e for e,c in cnt.items() if c/B>=t}
    # split half
    idx=(np.arange(len(X))+int(rng.integers(0,60)))//60
    h1,h2=run(X[idx%2==0])[0],run(X[idx%2==1])[0]
    print(f"seed{seed} true={len(tr)} single={len(e0)} prec={prec_rec(e0,tr)} b>=.5:{len(f(.5))} b>=.7:{len(f(.7))} prec.7={prec_rec(f(.7),tr) if f(.7) else None} b>=.9:{len(f(.9))} | split n1={len(h1)} n2={len(h2)} nshd={nshd(h1,h2):.2f}")
