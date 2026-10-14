"""Plot completed training/cleaning sensitivity analyses in the manuscript palette."""
from pathlib import Path
import argparse,json,sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.lines import Line2D
from pypdf import PdfReader,PdfWriter

CODE=Path(__file__).resolve().parent
PKG=CODE.parent
RESULTS_ROOT=PKG/'results'
FIGURES_ROOT=PKG/'figures'
sys.path.insert(0,str(CODE/'figure_tools'))
from ml_publication_style import configure_publication_style,style_axis,numeric_ticks,export_subfigures,save_figure_with_subfigures
from scientific_palettes import get_palette
from audit_palette_usage import audit_series_artists
PALETTE=get_palette('au_nimno_soft_8')
INK='#30343B'
DIRECT,ANCHORED=PALETTE[4],PALETTE[3]
CASES=['baseline','negative_missing','keep_extreme','source_quality','quality_negative_missing']
LABELS=['Original','Negative\nmissing','Keep 12','Source\nquality','Quality +\nnegative\nmissing']

def style(ax,xlabel='',ylabel='',numeric_x=True):
 style_axis(ax,xlabel=xlabel,ylabel=ylabel,tick_size=7,label_size=8)
 ax.tick_params(width=.65,length=3)
 for spine in ax.spines.values():spine.set_linewidth(.65)
 for tick in ax.xaxis.get_major_ticks()+ax.yaxis.get_major_ticks():
  tick.tick1line.set_markersize(3);tick.tick1line.set_markeredgewidth(.65)
 numeric_ticks(ax,x=numeric_x,y=True,n=4)
 ax.xaxis.labelpad=3;ax.yaxis.labelpad=3

def metadata(folder):
 for path in folder.rglob('*.pdf'):
  reader=PdfReader(path);writer=PdfWriter(clone_from=str(path))
  writer.add_metadata({'/Author':'CHONG LIU','/Creator':'CHONG LIU'})
  staged=path.with_suffix('.metadata.pdf')
  with staged.open('wb') as handle:writer.write(handle)
  staged.replace(path)
  check=PdfReader(path);assert check.metadata.author==check.metadata.creator=='CHONG LIU'

def save(fig,folder,stem,artists=None):
 if artists is not None:
  report=audit_series_artists('au_nimno_soft_8',artists)
  folder.mkdir(parents=True,exist_ok=True)
  (folder/'palette_audit.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
 save_figure_with_subfigures(fig,folder,stem,dpi=600,close=False)
 # Hide neighboring panels while cropping: their long labels can otherwise
 # enter another panel's expanded crop in a deliberately compact layout.
 axes=list(fig.axes)
 shared_legend=fig.legends[0] if fig.legends else None
 if shared_legend is not None:shared_legend.set_visible(False)
 for index,ax in enumerate(axes):
  for other in axes:other.set_visible(other is ax)
  local_legend=None
  if shared_legend is not None:
   handles=shared_legend.legend_handles
   labels=[entry.get_text() for entry in shared_legend.get_texts()]
   local_legend=ax.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,1.48),
    ncol=2,fontsize=6.3,handlelength=2,columnspacing=1,labelspacing=.3)
  letter='abcd'[index]
  isolated_stem=stem+'_'+letter
  export_subfigures(fig,folder,isolated_stem,dpi=600,pad=1.02)
  for suffix in ['png','pdf']:
   (folder/'subfigures'/f'{isolated_stem}_a.{suffix}').replace(folder/'subfigures'/f'{isolated_stem}.{suffix}')
  if local_legend is not None:local_legend.remove()
 for ax in axes:ax.set_visible(True)
 if shared_legend is not None:shared_legend.set_visible(True)
 lines=['# Subfigures','','Standalone exports from the final plotted data and styling.','']
 lines += [f'- {letter}: `{stem}_{letter}.png`, `{stem}_{letter}.pdf`' for letter in 'abcd']
 (folder/'subfigures/README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 plt.close(fig)
 metadata(folder)

def training(site='hofen'):
 assert site in ['hofen','avedore']
 is_hofen=site=='hofen'
 data=RESULTS_ROOT/('baseline_training_sensitivity' if is_hofen else 'avedore_training_sensitivity')
 curves=pd.read_csv(data/'learning_curves.csv')
 if 'stage' in curves:curves=curves[curves.stage=='checkpoint_selection']
 curves=curves[curves.seed==42].copy()
 summary=pd.read_csv(data/'summary.csv')
 architectures=['LSTM','GRU','TCN','Transformer']
 variants=[v for v in ['legacy','centered','raw_anchor'] if v in set(summary.variant)]
 expected={(a,v,f) for a in architectures for v in variants for f in ['direct','anchored']}
 completed={(r.architecture,r.variant,r.formulation) for r in summary.itertuples() if r.seed==42}
 assert expected<=completed,(expected-completed)
 figure='FigS21_training_sensitivity' if is_hofen else 'FigS23_avedore_training_sensitivity'
 folder=FIGURES_ROOT/'supplementary'/figure
 linestyle={'legacy':'-','centered':'--','raw_anchor':':'}
 fig,axes=plt.subplots(2,2,figsize=(7.0866,4.25))
 fig.subplots_adjust(left=.1,right=.99,bottom=.12,top=.79,wspace=.3,hspace=.36)
 artists={'Direct':[],'Anchored':[]}
 for letter,architecture,ax in zip('abcd',architectures,axes.flat):
  for variant in variants:
   for form,color in [('direct',DIRECT),('anchored',ANCHORED)]:
    rows=curves[(curves.architecture==architecture)&(curves.variant==variant)&(curves.formulation==form)&(curves.seed==42)].sort_values('epoch')
    assert len(rows) and np.isfinite(rows.validation_rmse).all()
    line,=ax.plot(rows.epoch,rows.validation_rmse,color=color,ls=linestyle[variant],lw=1.05,
      path_effects=[path_effects.Stroke(linewidth=1.55,foreground=INK,alpha=.55),path_effects.Normal()])
    artists[form.title()].append(line)
    best=rows[rows.is_best.astype(str).str.lower().eq('true')]
    assert len(best)==1
    mark=ax.scatter(best.epoch,best.validation_rmse,s=15,marker='o' if form=='direct' else 's',color=color,edgecolor=INK,linewidth=.35,zorder=4)
    artists[form.title()].append(mark)
  style(ax,'Epoch','Validation RMSE (g N h$^{-1}$)' if is_hofen else 'Validation RMSE (mg N L$^{-1}$)')
  observed=curves[(curves.architecture==architecture)&(curves.seed==42)].validation_rmse
  ax.set_ylim(bottom=max(0,float(observed.min())*.8))
  ax.text(0,1.04,letter,transform=ax.transAxes,weight='bold',fontsize=9,va='bottom')
  ax.text(.98,1.04,architecture,ha='right',va='bottom',transform=ax.transAxes,fontsize=8)
 names={'legacy':'Original scales','centered':'Centered output','raw_anchor':'Raw current-value input'}
 handles=[Line2D([],[],color=c,lw=1.1,ls=linestyle[v],marker='o' if f=='Direct' else 's',markersize=3.5,
   markeredgecolor=INK,markeredgewidth=.3,label=f'{f}: {names[v]}',
   path_effects=[path_effects.Stroke(linewidth=1.55,foreground=INK,alpha=.55),path_effects.Normal()])
   for f,c in [('Direct',DIRECT),('Anchored',ANCHORED)] for v in variants]
 for handle in handles:artists[handle.get_label().split(':')[0]].append(handle)
 fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.53,.995),ncol=2,fontsize=7,handlelength=2.0,columnspacing=1.2,labelspacing=.4)
 save(fig,folder,folder.name,artists)
 curves.to_csv(folder/'learning_curves_display.csv',index=False)
 print(('Fig. S21' if is_hofen else 'Fig. S23')+' exported with four panels.')

def cleaning():
 data=RESULTS_ROOT/'avedore_cleaning_sensitivity'
 info=json.loads((data/'summary.json').read_text(encoding='utf-8'))
 assert info['complete'] and info['case_count']==5
 coverage=pd.read_csv(data/'coverage_summary.csv').set_index('case').loc[CASES]
 metrics=pd.read_csv(data/'common_origin_benchmarks.csv')
 metrics=metrics[metrics.scope=='common_origins_own_target']
 design=pd.read_csv(data/'validation_design.csv')
 folder=FIGURES_ROOT/'supplementary/FigS22_cleaning_sensitivity'
 fig,axes=plt.subplots(2,2,figsize=(7.0866,4.65))
 fig.subplots_adjust(left=.1,right=.99,bottom=.17,top=.94,wspace=.32,hspace=.5)
 artists={'Training':[],'Test':[],'Direct':[],'Anchored':[],'Persistence':[], 'Held-out year':[],'Random hours':[]}
 x=np.arange(5);ax=axes[0,0]
 for offset,column,color,label in [(-.18,'n_train',PALETTE[0],'Training'),(.18,'n_test',PALETTE[1],'Test')]:
  bars=ax.bar(x+offset,coverage[column],width=.36,color=color,edgecolor=INK,linewidth=.3,label=label)
  artists[label].append(bars)
  for bar,value in zip(bars,coverage[column]):ax.text(bar.get_x()+bar.get_width()/2,value-180,f'{value:,}',ha='center',va='top',fontsize=5.6,rotation=90)
 ax.set_xticks(x,LABELS);ax.set_ylim(0,10700)
 style(ax,'','Forecast origins',numeric_x=False)
 ax.tick_params(axis='x',labelsize=6.5)
 ax.legend(loc='upper left',ncol=2,fontsize=6.5,handlelength=1)
 for ax,name in [(axes[0,1],'GRU'),(axes[1,0],'Transformer')]:
  for offset,model,color,label in [(-.17,name,DIRECT,'Direct'),(.17,name+'-A',ANCHORED,'Anchored')]:
   rows=metrics[metrics.model==model].set_index('scenario').loc[CASES]
   bars=ax.bar(x+offset,rows.R2,width=.34,color=color,edgecolor=INK,linewidth=.3,label=label)
   artists[label].append(bars)
  persistence=metrics[metrics.model=='Persistence'].set_index('scenario').loc[CASES]
  line,=ax.plot(x,persistence.R2,color=PALETTE[2],marker='_',markersize=10,linestyle='none',markeredgewidth=1.25,label='Persistence',zorder=4)
  artists['Persistence'].append(line)
  ax.axhline(0,color=INK,lw=.6)
  ax.set_xticks(x,LABELS)
  style(ax,'','Common-origin R²',numeric_x=False)
  lo=min(0,float(metrics[metrics.model.isin([name,name+'-A'])].R2.min()))-.08
  ax.set_ylim(lo,.82)
  ax.tick_params(axis='x',labelsize=6.5)
  ax.text(.98,1.035,name,ha='right',va='bottom',transform=ax.transAxes,fontsize=8)
  handles,labels=ax.get_legend_handles_labels()
  order=[labels.index(s) for s in ['Direct','Anchored','Persistence']]
  ax.legend([handles[k] for k in order],[labels[k] for k in order],loc='upper left',ncol=3,fontsize=6.3,handlelength=1,columnspacing=.8)
 ax=axes[1,1]
 for i,case in enumerate(CASES):
  rows=design[(design.scenario==case)&(design.model=='LightGBM')]
  chrono=rows[rows.design=='held_out_year'].Skill.iloc[0]
  random=rows[rows.design=='random_hours'].Skill.to_numpy();assert len(random)==5
  mean=random.mean()
  ax.plot([chrono,mean],[i,i],color='#C5C8CD',lw=.8)
  mark=ax.scatter([chrono],[i],s=24,color=PALETTE[0],edgecolor=INK,lw=.35,label='Held-out year' if i==0 else None,zorder=3)
  artists['Held-out year'].append(mark)
  err=ax.errorbar(mean,i,xerr=[[mean-random.min()],[random.max()-mean]],fmt='D',ms=4,color=PALETTE[1],mec=INK,mew=.35,lw=.8,capsize=2,label='Random hours' if i==0 else None)
  artists['Random hours'].extend([err.lines[0],*err.lines[1],*err.lines[2]])
 ax.axvline(0,color=INK,lw=.6)
 d_labels=['Original','Negative\nmissing','Keep 12','Source\nquality','Quality +\nnegative missing']
 ax.set_yticks(x,d_labels);ax.invert_yaxis()
 style(ax,'MSE skill vs persistence','',numeric_x=True)
 ax.yaxis.set_ticks(x,d_labels);ax.tick_params(axis='y',labelsize=6.5)
 ax.set_ylim(4.75,-.65)
 ax.legend(loc='upper center',bbox_to_anchor=(.5,1.12),ncol=2,fontsize=6.3,handlelength=1,columnspacing=.8)
 for letter,ax in zip('abcd',axes.flat):ax.text(-.17,1.035,letter,transform=ax.transAxes,weight='bold',fontsize=9)
 save(fig,folder,folder.name,artists)
 coverage.to_csv(folder/'coverage_display.csv')
 metrics[metrics.model.isin(['Persistence','GRU','GRU-A','Transformer','Transformer-A'])].to_csv(folder/'common_origin_metrics_display.csv',index=False)
 design[design.model=='LightGBM'].to_csv(folder/'validation_design_display.csv',index=False)
 print('Fig. S22 exported with four panels.')

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('which',choices=['training','avedore_training','cleaning','all'])
 parser.add_argument('--results-root',type=Path,default=RESULTS_ROOT)
 parser.add_argument('--outputs-root',type=Path,default=FIGURES_ROOT)
 args=parser.parse_args();RESULTS_ROOT=args.results_root.resolve();FIGURES_ROOT=args.outputs_root.resolve()
 configure_publication_style()
 plt.rcParams.update({'font.family':'Arial','font.size':7,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,'axes.linewidth':.65})
 if args.which in ['training','all']:training()
 if args.which in ['avedore_training','all']:training('avedore')
 if args.which in ['cleaning','all']:cleaning()
