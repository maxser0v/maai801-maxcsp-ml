"""Write readable optional-experiment results and a comparison figure."""
from pathlib import Path
import os,json
os.environ.setdefault('MPLCONFIGDIR','/tmp/maai801-model-search-mpl')
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parent
lock=json.loads((OUT/'selection_lock.json').read_text())
cv=pd.read_csv(OUT/'cv_leaderboard.csv')
summary=json.loads((OUT/'evaluation_summary.json').read_text())
rows={(r['dataset'],r['model']):r for r in summary}
fig,axs=plt.subplots(1,2,figsize=(11,4.6),layout='constrained')
for ax,dataset,title in zip(axs,['original_test_exploratory','fresh_test'],['Original test: 200 states (previously seen)','Fresh synthetic test: 2,500 states']):
 baseline=rows[dataset,'original_forest'];new=rows[dataset,'selection_champion']
 values=[baseline['heuristic_mean_gain'],baseline['mean_actual_gain'],new['mean_actual_gain']]
 bars=ax.bar(['Heuristic','Original forest','New SVR'],values,color=['#8a949e','#3977a8','#d27a35'])
 ax.set_ylim(0,2.1);ax.set_ylabel('Mean actual reduction in violations');ax.set_title(title);ax.grid(axis='y',alpha=.2)
 for b,v in zip(bars,values):ax.text(b.get_x()+b.get_width()/2,v+.025,f'{v:.4f}',ha='center',fontsize=10)
fig.savefig(OUT/'model_comparison.png',dpi=180);plt.close(fig)
family=cv.sort_values(['cv_mean_actual_gain','cv_mae'],ascending=[False,True]).drop_duplicates('family')
family.to_csv(OUT/'best_per_family_cv.csv',index=False)
text=(OUT/'report_text.md').read_text()
(OUT/'RESULTS.md').write_text(text)
print(OUT/'RESULTS.md')
