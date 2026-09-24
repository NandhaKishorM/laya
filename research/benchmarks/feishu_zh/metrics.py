"""Aggregate fixed first-repeat quality and all-repeat timing without dropping failures."""
import collections,math,statistics
LABELS=['urgent','todo','valuable','noise']
def percentile(values,q):
 if not values:return None
 xs=sorted(values);p=(len(xs)-1)*q;lo=math.floor(p);hi=math.ceil(p)
 return xs[lo]+(xs[hi]-xs[lo])*(p-lo)
def quality(rows):
 n=len(rows);ok=[r for r in rows if r['status']=='ok'];correct=sum(r.get('predicted')==r['expected'] for r in rows)
 matrix={a:{b:0 for b in LABELS+['ERROR']} for a in LABELS}
 for r in rows:matrix[r['expected']][r.get('predicted','ERROR')]+=1
 f1=[]
 for label in LABELS:
  tp=matrix[label][label];fp=sum(matrix[x][label] for x in LABELS if x!=label);fn=sum(v for x,v in matrix[label].items() if x!=label)
  f1.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0)
 nonaction=[r for r in rows if r['expected'] in ['valuable','noise']];action=[r for r in rows if r['expected'] in ['urgent','todo']]
 urgent=[r for r in rows if r['expected']=='urgent']
 return {'n':n,'correct':correct,'accuracy':correct/n if n else None,'errors':n-len(ok),'macro_f1':statistics.mean(f1),'false_action_count':sum(r.get('predicted') in ['urgent','todo'] for r in nonaction),'nonaction_denominator':len(nonaction),'missed_action_count':sum(r.get('predicted') not in ['urgent','todo'] for r in action),'action_denominator':len(action),'urgent_recalled':sum(r.get('predicted')=='urgent' for r in urgent),'urgent_denominator':len(urgent),'confusion_matrix':matrix}
def summarize(rows,expected_repeats=3):
 first=[r for r in rows if r['repeat']==0];result=quality(first);success=[r for r in rows if r['status']=='ok'];times=[r['elapsed_ms'] for r in success]
 result['timing']={'n':len(times),'p50_ms':percentile(times,.5),'p95_ms':percentile(times,.95),'mean_ms':statistics.mean(times) if times else None,'definition':'Client end-to-end wall clock, excluding warmup; p95 uses linear interpolation.'}
 result['total_requests']=len(rows);result['failed_requests']=len(rows)-len(success)
 grouped=collections.defaultdict(list)
 for r in rows:grouped[r['id']].append(r)
 complete=[v for v in grouped.values() if len(v)==expected_repeats and all(r['status']=='ok' for r in v)]
 result['repeat_consistency']={'complete_cases':len(complete),'same_label_all_three' if expected_repeats==3 else 'same_label_all_repeats':sum(len({r['predicted'] for r in v})==1 for v in complete)}
 result['by_family']={family:quality([r for r in first if r['family']==family]) for family in sorted({r['family'] for r in first})}
 result['truncation']={key:sum(r['diagnostics'][key] for r in first) if all(key in r.get('diagnostics',{}) for r in first) else None for key in ['state_truncated','instructions_truncated']}
 if any('lengths' in r.get('diagnostics',{}) for r in first):result['quality_without_truncation']=quality([r for r in first if r['status']=='ok' and not r['diagnostics']['state_truncated'] and not r['diagnostics']['instructions_truncated']])
 choice=[r for r in first if r['status']=='ok' and 'probabilities' in r]
 if choice:
  brier=statistics.mean(sum((r['probabilities'][label]-(label==r['expected']))**2 for label in LABELS) for r in choice)
  bins=[[] for _ in range(10)]
  for r in choice:
   conf=r['probabilities'][r['predicted']];bins[min(9,int(conf*10))].append((conf,r['predicted']==r['expected']))
  ece=sum(len(b)/len(choice)*abs(statistics.mean(x for x,y in b)-statistics.mean(y for x,y in b)) for b in bins if b)
  result['choice_probability_diagnostics']={'n':len(choice),'multiclass_brier_sum':brier,'ece_10_equal_width_bins':ece,'caution':'Descriptive only; 64 synthetic labels, no calibration fitting or general calibration claim.'}
 return result
