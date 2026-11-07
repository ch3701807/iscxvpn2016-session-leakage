"""v2-compatible streaming feature extraction with filtering and visible TLS SNI masking."""
import argparse,bisect,csv,hashlib,json,re,sys,time
from pathlib import Path
from collections import defaultdict,Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from pcap_stream import Capture,ipv4_transport as fast_transport

def ipv4_transport(data,link):
    result=fast_transport(data,link)
    # Preserve v2 Scapy serialization (notably ASN.1 SNMP) and zero IP.len offload records.
    needs_scapy=(result is not None and (result[2] in PORTS or result[3] in PORTS))
    if result is None:
        needs_scapy=(link==1 and len(data)>=34 and data[12:14]==b"\x08\x00")
    if not needs_scapy:return result
    from scapy.all import conf,IP,TCP,UDP
    pkt=conf.l2types[link](data)
    trans=pkt[TCP] if TCP in pkt else (pkt[UDP] if UDP in pkt else None)
    if IP not in pkt or trans is None:return None
    ip=pkt[IP]
    return (tuple(map(int,ip.src.split("."))),tuple(map(int,ip.dst.split("."))),int(trans.sport),int(trans.dport),int(ip.proto),int(ip.len),bytes(trans.payload),link==1 and data[:6]==b"\xff"*6,int(trans.seq) if TCP in pkt else 0,int(trans.ack) if TCP in pkt else 0,int(trans.flags) if TCP in pkt else 0)
W=Path(__file__).resolve().parents[1]
PORTS={53,5353,5355,137,138,139,1900,67,68,546,547,123,161,162,3702}
KEYS=['seq','iat','raw','stat','label']

def dump(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)

def session(name):return re.sub(r'_?[aAbB]$','',Path(name).stem)

def mask_sni(payload):
    """Parse visible TLS records/ClientHello fields; never infer an unseen hostname."""
    buf=bytearray(payload);parsed=False;sni=False;changed=0;pos=0
    while pos+5<=len(buf):
        typ,major=buf[pos],buf[pos+1]
        if major!=3 or typ not in [20,21,22,23]:break
        record_end=pos+5+int.from_bytes(buf[pos+3:pos+5],'big')
        if typ==22:
            hp=pos+5
            while hp+4<=min(record_end,len(buf)):
                htyp=buf[hp];he=hp+4+int.from_bytes(buf[hp+1:hp+4],'big')
                limit=min(he,record_end,len(buf))
                if htyp==1:
                    p=hp+4
                    if p+35<=limit:
                        p+=34;sid=buf[p];p+=1+sid
                        if p+2<=limit:
                            clen=int.from_bytes(buf[p:p+2],'big');p+=2+clen
                            if p+1<=limit:
                                comp=buf[p];p+=1+comp
                                if p<=limit:
                                    parsed=True
                                    if p+2<=limit:
                                        ext_end=p+2+int.from_bytes(buf[p:p+2],'big');p+=2
                                        while p+4<=min(ext_end,limit):
                                            kind=int.from_bytes(buf[p:p+2],'big');n=int.from_bytes(buf[p+2:p+4],'big');ep=p+4+n
                                            if ep>min(ext_end,limit):break
                                            if kind==0 and p+6<=ep:
                                                names_end=p+6+int.from_bytes(buf[p+4:p+6],'big');q=p+6
                                                if names_end<=ep:
                                                    while q+3<=names_end:
                                                        nt=buf[q];nn=int.from_bytes(buf[q+1:q+3],'big');ns=q+3;ne=ns+nn
                                                        if ne>names_end:break
                                                        if nt==0 and nn:
                                                            sni=True;changed+=sum(x!=0 for x in buf[ns:ne]);buf[ns:ne]=b'\0'*nn
                                                        q=ne
                                            p=ep
                if he>record_end or he>len(buf):break
                hp=he
        if record_end>len(buf):break
        pos=record_end
    return bytes(buf),int(parsed),int(sni),changed

def build_row(pkts):
    times=np.array([p[0] for p in pkts],dtype=np.float64)
    iat=np.zeros(50,dtype=np.float32);iat[1:len(pkts)]=np.diff(times)
    seq=np.zeros(50,dtype=np.float32);seq[:len(pkts)]=[p[1]*p[2] for p in pkts]
    payload=b''.join(p[3] for p in pkts)[:256]
    raw=np.zeros(256,dtype=np.float32);raw[:len(payload)]=np.frombuffer(payload,dtype=np.uint8)/255.
    masked,ch,sni,nb=mask_sni(payload)
    raw_mask=np.zeros(256,dtype=np.float32);raw_mask[:len(masked)]=np.frombuffer(masked,dtype=np.uint8)/255.
    a=np.abs(seq[:len(pkts)])
    stat=np.array([a.mean(),a.std(),a.max(),times[-1]-times[0],len(pkts),iat[1:len(pkts)].mean()],dtype=np.float32)
    return seq,iat,raw,stat,raw_mask,(ch,sni,nb)

def worker(spec):
    name,cname,cid,muse=spec;muse=Path(muse)
    output=W/'data/v3_files'/f'{Path(name).stem}.npz'
    done=W/'markers/extract_files'/f'{name}.json'
    if done.exists() and output.exists():return json.loads(done.read_text())
    flows={};r=Capture(muse/'data/ISCXVPN2016'/name);n=0;first=last=lo=hi=None;ip_strings={};counts=Counter()
    for t,data,link in r:
        n+=1
        if first is None:first=lo=hi=t
        last=t;lo=min(lo,t);hi=max(hi,t)
        p=ipv4_transport(data,link)
        if p is None:continue
        src,dst,sport,dport,proto,total,payload,bcast,_,_,_=p
        if src not in ip_strings:ip_strings[src]='.'.join(map(str,src))
        if dst not in ip_strings:ip_strings[dst]='.'.join(map(str,dst))
        ep1=(ip_strings[src],sport);ep2=(ip_strings[dst],dport)
        key=(min(ep1,ep2),max(ep1,ep2),proto)
        chain=flows.setdefault(key,[])
        if not chain or t-chain[-1]['last']>120:
            chain.append({'initiator':src,'pkts':[],'last':t,'drop':False,'reasons':set()})
        f=chain[-1];f['last']=t
        reasons=[]
        if sport in PORTS or dport in PORTS:reasons.append('standard_background_port')
        if 224<=dst[0]<=239:reasons.append('destination_multicast')
        if dst==(255,255,255,255) or bcast:reasons.append('destination_broadcast')
        if reasons:f['drop']=True;f['reasons'].update(reasons)
        pkt=(t,1 if src==f['initiator'] else -1,min(total,1500),payload)
        bisect.insort_right(f['pkts'],pkt)
        if len(f['pkts'])>50:f['pkts'].pop()
    raw_sha=r.sha.hexdigest();header=r.header;r.close()
    seqs=[];iats=[];raws=[];stats=[];masks=[];keep=[];snim=[];ids=[]
    for key in sorted(flows):
        for f in flows[key]:
            if len(f['pkts'])<5:continue
            seq,iat,raw,stat,masked,sni=build_row(f['pkts'])
            ids.append(hashlib.sha256((name+':v3:'+str(len(ids))).encode()).hexdigest()[:24])
            seqs.append(seq);iats.append(iat);raws.append(raw);stats.append(stat);masks.append(masked);keep.append(not f['drop']);snim.append(sni)
            if f['drop']:counts.update(f['reasons'])
    N=len(seqs)
    arr=lambda xs,d:np.stack(xs) if xs else np.zeros((0,d),dtype=np.float32)
    data={'seq':arr(seqs,50),'iat':arr(iats,50),'raw':arr(raws,256),'stat':arr(stats,6),'raw_masked':arr(masks,256),'label':np.full(N,cid,dtype=np.int64),'keep_clean':np.array(keep,dtype=bool),'sni_counts':np.array(snim,dtype=np.int64).reshape(N,3),'flow_id':np.array(ids,dtype='U24')}
    reference=np.load(muse/'data/iscx_v2'/f'{Path(name).stem}.npz')
    diffs={}
    for k in KEYS:
        if not np.array_equal(data[k],reference[k]):
            diffs[k]={'shape':list(data[k].shape),'reference_shape':list(reference[k].shape)}
            if data[k].shape==reference[k].shape:diffs[k]['mismatched_elements']=int(np.count_nonzero(data[k]!=reference[k]))
    if diffs:
        dump(W/'audit/extraction_mismatches'/f'{name}.json',{'file':name,'differences':diffs})
        raise RuntimeError('v2 feature equivalence check failed for '+name+'; see audit/extraction_mismatches')
    output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,**data)
    before=np.array(snim,dtype=np.int64).reshape(N,3).sum(0)
    after=data['sni_counts'][data['keep_clean']].sum(0)
    result={'file':name,'class':cname,'class_id':cid,'session':session(name),'packet_count':n,'first_packet_timestamp':first,'last_packet_timestamp':last,'min_packet_timestamp':lo,'max_packet_timestamp':hi,'header':header,'before_flows':N,'after_flows':int(sum(keep)),'filter_reasons':dict(counts),'sni_before_clean':before.tolist(),'sni_after_clean':after.tolist(),'input_sha256':raw_sha,'v2_all_arrays_equal':True}
    dump(done,result);return result

def main():
    registration=json.loads((W/'audit/rules_registration.json').read_text())
    assert hashlib.sha256((W/'RULES.md').read_bytes()).hexdigest()==registration['sha256']
    muse=Path(json.loads((W/'config/paths.local.json').read_text())['muse_root'])
    classes=json.loads((muse/'data/iscx_classes_v2.json').read_text())
    mapping=list(csv.DictReader((W/'audit/file_to_suffix_session.csv').open()))
    records=[];summaries=[]
    for cname,cid in sorted(classes.items(),key=lambda x:x[1]):
        specs=[(x['file'],cname,cid,str(muse)) for x in mapping if x['class']==cname]
        print('CLASS START',cname,len(specs),flush=True)
        rec=[];fail=[]
        with ProcessPoolExecutor(max_workers=12) as pool:
            futures={pool.submit(worker,s):s for s in specs}
            for future in as_completed(futures):
                spec=futures[future]
                try:result=future.result()
                except Exception as first:
                    print('RETRY ONCE',spec[0],str(first),flush=True)
                    dump(W/'audit/retries'/f'extract_{spec[0]}.json',{'file':spec[0],'first_error':str(first),'settings_changed':False,'retry_count':1})
                    try:result=pool.submit(worker,spec).result()
                    except Exception as second:
                        fail.append({'file':spec[0],'error':str(second)});continue
                rec.append(result)
                print('FILE DONE',result['file'],result['before_flows'],result['after_flows'],flush=True)
        records.extend(rec)
        if fail:
            dump(W/'audit/extraction_failed_files.json',fail)
            dump(W/'markers/STOPPED.json',{'reason':'Complete class data unavailable after original-setting retry; excluding failed files would alter the protocol','stage':'extraction','failed_files':fail})
            return 2
        before=sum(x['before_flows'] for x in rec);after=sum(x['after_flows'] for x in rec)
        bs=sorted({x['session'] for x in rec if x['before_flows']});cs=sorted({x['session'] for x in rec if x['after_flows']})
        summary={'class':cname,'class_id':cid,'files':[x['file'] for x in sorted(rec,key=lambda x:x['file'])],'file_count':len(rec),'before_flows':before,'after_flows':after,'removed_fraction':(before-after)/before if before else 0,'before_sessions':len(bs),'after_sessions':len(cs),'session_names_after_clean':cs,'sni_before_clean':np.array([x['sni_before_clean'] for x in rec]).sum(0).tolist(),'sni_after_clean':np.array([x['sni_after_clean'] for x in rec]).sum(0).tolist()}
        summaries.append(summary)
        dump(W/'audit/v3_cleaning_inventory.json',{'complete':len(summaries)==len(classes),'classes':summaries,'files':records})
        (W/'audit/RAW_INPUT_SHA256SUMS').write_text(''.join(x['input_sha256']+'  muse/data/ISCXVPN2016/'+x['file']+'\n' for x in sorted(records,key=lambda x:x['file'])))
        print('CLASS DONE',json.dumps(summary,ensure_ascii=False),flush=True)
        if before and after<before*.5:
            dump(W/'markers/STOPPED.json',{'reason':'class loses more than 50 percent of flows under required background cleaning','stage':'cleaning_gate','class':cname,'before_flows':before,'after_flows':after,'removed_fraction':(before-after)/before,'processed_classes':len(summaries),'all_classes':len(classes),'rule_unchanged':True})
            return 3
    kept=[x['class'] for x in summaries if x['after_sessions']>=2]
    excluded=[x for x in summaries if x['after_sessions']<2]
    dump(W/'config/final_classes.json',{'main_classes':kept,'excluded':excluded,'fixed_for_all_conditions':True,'binary_keeps_all_classes':True})
    if len(kept)<8:
        dump(W/'markers/STOPPED.json',{'reason':'fewer than eight main classes after cleaning and session exclusion','main_classes':kept});return 4
    pcaps=sorted(records,key=lambda x:(Path(x['file']).suffix!='.pcap',x['file']))
    merged=defaultdict(list)
    for record in pcaps:
        d=np.load(W/'data/v3_files'/f'{Path(record["file"]).stem}.npz')
        for key in d.files:merged[key].append(d[key])
        N=len(d['label']);merged['source_file'].append(np.array([record['file']]*N));merged['session'].append(np.array([record['session']]*N));merged['vpn_label'].append(np.full(N,int(record['file'].startswith('vpn_')),dtype=np.int64))
    np.savez_compressed(W/'data/iscx_v3.npz',**{k:np.concatenate(v) for k,v in merged.items()})
    dump(W/'markers/EXTRACTION_COMPLETED.json',{'classes':kept,'files':len(records),'flows':sum(x['before_flows'] for x in records),'clean_flows':sum(x['after_flows'] for x in records)})
    return 0
if __name__=='__main__':sys.exit(main())
