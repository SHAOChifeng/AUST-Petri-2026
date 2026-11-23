#!/usr/bin/env python3
"""
Figure 11 (real): Algorithm 3 low-frequency frost prediction example.
Uses a REAL LSTM 7-day forecast (2023-12-22 origin) from predictions.npz and
REAL seasonal climatology P(WS<3), P(RH>60) from the NASA POWER training set.
"""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","DejaVu Sans"],
                     "font.size":9,"savefig.dpi":300,"figure.dpi":300})
FIG = Path(__file__).parent / "figures"
LSTM = Path(__file__).parent.parent / "lstm_agri"

# ── load real LSTM forecast ──
p = np.load(LSTM/"models/predictions.npz", allow_pickle=True)
cols=list(p["cols"]); vi=cols.index("T2M")
preds=p["preds"][:,:,vi]; T=pd.to_datetime(p["T_test"])
i=int(np.where(preds.min(axis=1)<0)[0][np.argmin(preds.min(axis=1)[preds.min(axis=1)<0])])
temp=preds[i]                         # real 7-day LSTM forecast
dates=[T[i]+pd.Timedelta(days=k) for k in range(7)]

# ── real climatology per calendar day from TRAIN ──
tr=pd.read_csv(LSTM/"data/processed/train.csv", parse_dates=["Date"]); tr["md"]=tr["Date"].dt.strftime("%m-%d")
g=tr.groupby("md")
pw=g["WS2M"].apply(lambda s:(s<3).mean()); pr=g["RH2M"].apply(lambda s:(s>60).mean())
mds=[d.strftime("%m-%d") for d in dates]
windp=np.array([pw.get(m,np.nan) for m in mds]); rhp=np.array([pr.get(m,np.nan) for m in mds])

days=np.arange(1,8)
t_ok=temp<0; w_ok=windp>0.7; h_ok=rhp>0.7
inter=t_ok&w_ok&h_ok
first=int(np.argmax(inter)) if inter.any() else None

C={"t":"#EA4335","w":"#1A73E8","h":"#34A853","i":"#9334E6","gray":"#5F6368"}
fig,ax=plt.subplots(4,1,figsize=(7,7.2),sharex=True)

ax[0].plot(days,temp,"-o",color=C["t"],lw=1.8,ms=4)
ax[0].axhline(0,ls="--",color=C["t"],alpha=0.6,lw=1)
ax[0].fill_between(days,0,temp,where=t_ok,color=C["t"],alpha=0.15)
ax[0].set_ylabel("Temperature (°C)",fontsize=8)
ax[0].set_title("(a) LSTM 7-day temperature forecast (real, 2023-12-22 origin); threshold 0 °C",fontsize=8,fontweight="bold")
for d,tv in zip(dates,temp):
    ax[0].annotate(f"{tv:.1f}",(days[list(dates).index(d)],tv),textcoords="offset points",xytext=(0,6),ha="center",fontsize=6,color=C["t"])

ax[1].bar(days,windp,color=C["w"],alpha=0.8,width=0.55)
ax[1].axhline(0.7,ls="--",color=C["w"],alpha=0.7,lw=1)
ax[1].set_ylabel("P(WS < 3 m/s)",fontsize=8); ax[1].set_ylim(0,1.05)
ax[1].set_title("(b) Climatological probability of low wind (threshold 0.7)",fontsize=8,fontweight="bold")

ax[2].bar(days,rhp,color=C["h"],alpha=0.8,width=0.55)
ax[2].axhline(0.7,ls="--",color=C["h"],alpha=0.7,lw=1)
ax[2].set_ylabel("P(RH > 60%)",fontsize=8); ax[2].set_ylim(0,1.05)
ax[2].set_title("(c) Climatological probability of high humidity (threshold 0.7)",fontsize=8,fontweight="bold")

ax[3].fill_between(days,0,1.2,where=inter,color=C["i"],alpha=0.45,step="mid")
ax[3].plot(days,inter.astype(int),"-o",color=C["i"],lw=1.6,ms=4,drawstyle="steps-mid")
ax[3].set_ylim(-0.1,1.35); ax[3].set_yticks([0,1]); ax[3].set_yticklabels(["No","Yes"],fontsize=7)
ax[3].set_ylabel("Frost risk",fontsize=8); ax[3].set_xlabel("Forecast day",fontsize=9)
ax[3].set_xticks(days); ax[3].set_xticklabels([f"Day {d}\n{dt.strftime('%m-%d')}" for d,dt in zip(days,dates)],fontsize=7)
ax[3].set_title("(d) Precursor intersection → predicted frost window",fontsize=8,fontweight="bold")
if first is not None:
    ax[3].annotate(f"Earliest joint frost\n= Day {first+1} ({dates[first].strftime('%Y-%m-%d')})",
                   xy=(first+1,1),xytext=(first+1+2.2,0.55),fontsize=7,color=C["i"],fontweight="bold",
                   arrowprops=dict(arrowstyle="->",color=C["i"],lw=1))
for a in ax:
    a.grid(axis="y",alpha=0.15); a.tick_params(direction="in",length=3)
plt.tight_layout()
fig.savefig(FIG/"fig11_frost_intersection.png", bbox_inches="tight"); fig.savefig(FIG/"fig11_frost_intersection.svg", bbox_inches="tight")
print("saved. first frost day:", None if first is None else (first+1, dates[first].date()))
print("temp:",np.round(temp,2)); print("windp:",np.round(windp,2)); print("rhp:",np.round(rhp,2)); print("inter:",inter.astype(int))
