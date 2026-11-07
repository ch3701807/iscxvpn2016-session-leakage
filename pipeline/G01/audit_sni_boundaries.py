"""Independent read-only check for visible but incompletely captured SNI names."""
import os,json
from pathlib import Path
import numpy as np
os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:16])
W=Path(__file__).resolve().parents[1]
data=np.load(W/'data/iscx_v3.npz');raw=data['raw'];clean=data['keep_clean'];ids=data['flow_id']
partial=[];visible_sni_names=0
for i in np.where(clean)[0]:
    b=np.rint(raw[i]*255).astype(np.uint8).tobytes()
    if len(b)<44 or b[0]!=22 or b[1]!=3 or b[5]!=1:continue
    recend=5+int.from_bytes(b[3:5],'big');hend=9+int.from_bytes(b[6:9],'big');limit=min(recend,hend,len(b));p=43
    if p>=limit:continue
    p+=1+b[p]
    if p+2>limit:continue
    p+=2+int.from_bytes(b[p:p+2],'big')
    if p+1>limit:continue
    p+=1+b[p]
    if p+2>limit:continue
    ext_end=p+2+int.from_bytes(b[p:p+2],'big');p+=2
    while p+4<=min(ext_end,limit):
        typ=int.from_bytes(b[p:p+2],'big');n=int.from_bytes(b[p+2:p+4],'big');ep=p+4+n
        if typ==0 and p+6<=limit:
            names_end=p+6+int.from_bytes(b[p+4:p+6],'big');q=p+6
            while q+3<=min(names_end,limit):
                nt=b[q];nn=int.from_bytes(b[q+1:q+3],'big');ns=q+3;ne=ns+nn
                if nt==0 and nn and ns<limit:
                    visible_sni_names+=1
                    if ne>limit:partial.append({'flow_id':str(ids[i]),'visible_name_bytes':limit-ns,'declared_name_bytes':nn})
                q=ne
        if ep>limit:break
        p=ep
result={'clean_flows':int(clean.sum()),'visible_sni_name_records_found':visible_sni_names,'partially_captured_sni_names_with_visible_bytes':len(partial),'partial_records':partial}
(W/'audit/sni_visible_boundary_validation.json').write_text(json.dumps(result,indent=2)+'\n')
print({k:v for k,v in result.items() if k!='partial_records'})
