"""Two phone-readable 3:4 cards, generated from the frozen result summary."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,Rectangle
from matplotlib.font_manager import FontProperties
ROOT=Path(__file__).resolve().parent
DPI=160; W,H=1440,1920
FONT=next((p for p in [Path('/System/Library/Fonts/Hiragino Sans GB.ttc'),Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')] if p.exists()),None)
LANG='zh'
NAVY='#142d40';MUTED='#61798a';BLUE='#2874c8';TEAL='#159b8e';BG='#f4f8fb'
S=json.loads((ROOT/'results/v1/summary.json').read_text(encoding='utf-8'))
ROWS=[S[b][m] for b,m in [('jev','choice'),('jev','four_noul'),('laya','choice'),('laya','four_noul')]]
NAMES=['Jev · 单选','Jev · 四问','Laya · 单选','Laya · 四问'];COLORS=[BLUE,BLUE,TEAL,TEAL]
def canvas():
 fig=plt.figure(figsize=(W/DPI,H/DPI),dpi=DPI,facecolor=BG);ax=fig.add_axes([0,0,1,1]);ax.set_xlim(0,W);ax.set_ylim(H,0);ax.axis('off');return fig,ax
EN={'Feishu 消息分类实测': 'Feishu message classification', '闭源分类器 API 与开源本地部署实测对比': 'Closed-source API vs open-source local deployment', '64 个合成场景  ·  8 类情境  ·  768 次计时请求': '64 synthetic cases  ·  8 scenario families  ·  768 timed requests', '分类准确率 ↑': 'Classification accuracy ↑', '越高越好': 'Higher is better', '误生成任务数 ↓': 'False task assignments ↓', '越少越好': 'Lower is better', '请求耗时 ↓': 'Request latency ↓', '越低越好': 'Lower is better', 'Jev · 单选': 'Jev · choice', 'Jev · 四问': 'Jev · 4Q', 'Laya · 单选': 'Laya · choice', 'Laya · 四问': 'Laya · 4Q', '虚线：始终猜同一类别的 25% 基线': 'Dashed line: 25% constant-class baseline', '32 条无需行动的消息，被误判为待办或紧急': 'Out of 32 non-action messages, mislabeled as tasks or urgent', '柱长：中位数 p50  ·  横线：p95（不是置信区间）': 'Bar: median p50  ·  whisker: p95 (not a confidence interval)', '准确率取首次冻结结果；三次重复全部公开。': 'Accuracy: first frozen repeat; all three repeats are published.', '每种配置计时 192 次，不含预热。': '192 timed requests per configuration, excluding warmup.', 'Laya 多语言版：M4 GPU 本地；Jev 1.13.0：API，含网络。': 'Laya multilingual: M4 GPU locally. Jev 1.13.0: API incl. network.', '这是不同部署方式的比较，并非同硬件测试。': 'Different deployment paths; not a same-hardware comparison.', 'AI 辅助合成诊断集，不代表真实业务总体准确率或通用能力排名。': 'AI-assisted synthetic diagnostic; not real-world accuracy or a general ranking.', '单选与四问的含义、完整中文表格 → 下一张': 'Choice / 4Q definitions and the full table → README', '数据与复现：Adkid-Zephyr / chinese-workflow-decision-bench': 'Data & code: Adkid-Zephyr / chinese-workflow-decision-bench'}
def text(ax,x,y,value,size=38,color=NAVY,ha='left'):
 if LANG=='en':
  value=EN.get(value,value).replace(' 毫秒',' ms')
  font=FontProperties(family='DejaVu Sans',size=size*72/DPI)
 else:
  if FONT is None:raise RuntimeError('Install Hiragino Sans GB or Noto Sans CJK.')
  font=FontProperties(fname=str(FONT),size=size*72/DPI)
 return ax.text(x,y,value,fontproperties=font,color=color,va='top',ha=ha)
def card(ax,y,height):ax.add_patch(FancyBboxPatch((60,y),1320,height,boxstyle='round,pad=0,rounding_size=25',facecolor='white',edgecolor='#e0e9f0',linewidth=1))
def header(ax,page):
 text(ax,80,60,'Feishu 消息分类实测',38,BLUE);text(ax,1360,67,page+' / 02',30,MUTED,ha='right')
 text(ax,80,124,'Jev vs Laya',96)
 text(ax,80,242,'闭源分类器 API 与开源本地部署实测对比',42)
 text(ax,80,322,'64 个合成场景  ·  8 类情境  ·  768 次计时请求',32,MUTED)
def save(fig,name):
 for ext in ['png','svg']:fig.savefig(ROOT/f'assets/{name}.{ext}',dpi=DPI,facecolor=BG,metadata={'Creator':'Feishu Message Classification Bench'})
 plt.close(fig)
def scorecard():
 fig,ax=canvas();header(ax,'01')
 specs=[(390,350,'分类准确率 ↑','越高越好','accuracy'),(765,350,'误生成任务数 ↓','越少越好','false'),(1140,365,'请求耗时 ↓','越低越好','latency')]
 for top,height,title,hint,metric in specs:
  card(ax,top,height);text(ax,95,top+24,title,46);text(ax,1340,top+35,hint,30,MUTED,ha='right')
  for i,(r,name,color) in enumerate(zip(ROWS,NAMES,COLORS)):
   y=top+105+i*53;text(ax,100,y-18,name,34)
   if metric=='accuracy':value=r['accuracy'];label=f"{value*100:.2f}% · {r['correct']}/64"
   elif metric=='false':value=r['false_action_count']/32;label=f"{r['false_action_count']}/32"
   else:value=r['timing']['p50_ms']/620;label=f"{r['timing']['p50_ms']:.0f} 毫秒"
   ax.add_patch(Rectangle((380,y-13),610,29,facecolor='#f0f4f7',edgecolor='none'))
   ax.add_patch(Rectangle((380,y-13),610*value,29,facecolor=color,edgecolor='none'))
   if metric=='accuracy':ax.plot([380+610*.25]*2,[y-18,y+21],color='#94a6b4',lw=1,ls='--')
   if metric=='latency':
    hi=380+610*r['timing']['p95_ms']/620;lo=380+610*value
    ax.plot([lo,hi],[y+2,y+2],color=NAVY,lw=1);ax.plot([hi,hi],[y-6,y+10],color=NAVY,lw=1)
   text(ax,1020,y-19,label,32)
  captions={'accuracy':'虚线：始终猜同一类别的 25% 基线','false':'32 条无需行动的消息，被误判为待办或紧急','latency':'柱长：中位数 p50  ·  横线：p95（不是置信区间）'}
  text(ax,100,top+height-43,captions[metric],27,MUTED)
 footer=['准确率取首次冻结结果；三次重复全部公开。','每种配置计时 192 次，不含预热。','Laya 多语言版：M4 GPU 本地；Jev 1.13.0：API，含网络。','这是不同部署方式的比较，并非同硬件测试。','AI 辅助合成诊断集，不代表真实业务总体准确率或通用能力排名。']
 for i,line in enumerate(footer):text(ax,80,1548+i*48,line,30,MUTED)
 text(ax,80,1812,'单选与四问的含义、完整中文表格 → 下一张',34,BLUE)
 text(ax,80,1870,'数据与复现：Adkid-Zephyr / chinese-workflow-decision-bench',24,MUTED)
 save(fig,'xiaohongshu-scorecard-3x4'+('-en' if LANG=='en' else ''))
def tablecard():
 fig,ax=canvas();header(ax,'02')
 for mode,title,top in [('choice','单选择题：直接四选一',390),('four_noul','四问组合：先判断，再按规则分类',840)]:
  card(ax,top,420);text(ax,95,top+22,title,44)
  y=top+90
  for x,width,color in [(95,345,'#edf2f6'),(440,440,'#edf5fd'),(880,460,'#eaf7f4')]:ax.add_patch(Rectangle((x,y),width,275,facecolor=color,edgecolor='white',lw=1))
  text(ax,120,y+12,'指标',34);text(ax,660,y+12,'Jev',38,BLUE,ha='center');text(ax,1110,y+12,'Laya 多语言版',38,TEAL,ha='center')
  a,b=S['jev'][mode],S['laya'][mode]
  records=[('正确分类',f"{a['correct']}/64 · {a['accuracy']*100:.2f}%",f"{b['correct']}/64 · {b['accuracy']*100:.2f}%"),('误生成任务',f"{a['false_action_count']}/32",f"{b['false_action_count']}/32"),('耗时中位数',f"{a['timing']['p50_ms']:.0f} 毫秒",f"{b['timing']['p50_ms']:.0f} 毫秒"),('p95 耗时',f"{a['timing']['p95_ms']:.0f} 毫秒",f"{b['timing']['p95_ms']:.0f} 毫秒")]
  for i,(label,av,bv) in enumerate(records):
   yy=y+68+i*49;text(ax,120,yy,label,32);text(ax,660,yy,av,34,ha='center');text(ax,1110,yy,bv,34,ha='center')
  text(ax,100,top+376,'误生成任务：把无需行动的消息判成待办或紧急。',27,MUTED)
 text(ax,80,1310,'四问分别判断什么？',43)
 text(ax,80,1380,'① 与我相关吗？     ② 需要我行动吗？',37)
 text(ax,80,1440,'③ 是否紧急？         ④ 是否有资料价值？',37)
 text(ax,80,1515,'测试口径',40)
 notes=['64 个合成案例；准确率取预先固定的首次结果。','每种配置重复三轮，共 192 次计时请求，排除预热。','Laya 多语言版：M4 GPU；Jev 1.13.0：API，含网络。','不同硬件与部署方式，不能当作同硬件速度对比。','小样本合成诊断，不代表真实业务总体准确率。']
 for i,line in enumerate(notes):text(ax,80,1580+i*47,line,31,MUTED)
 text(ax,80,1870,'数据与复现：Adkid-Zephyr / chinese-workflow-decision-bench',24,MUTED)
 save(fig,'xiaohongshu-table-3x4')
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--lang',choices=['zh','en'],default='zh');LANG=parser.parse_args().lang
 scorecard()
 if LANG=='zh':tablecard()
