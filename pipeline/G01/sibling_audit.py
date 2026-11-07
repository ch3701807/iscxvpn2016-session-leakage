"""CPU-only sibling evidence under unchanged, preregistered RULES.md."""
import csv,functools,gzip,hashlib,hmac,itertools,json,os,pickle,struct,sys
from pathlib import Path
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from pcap_stream import Capture,ipv4_transport
from extract_v3 import dump,session
W=Path(__file__).resolve().parents[1]
PRIVATE=W/'data/sibling_private'

def parsed_packet(data,link):
    p=ipv4_transport(data,link)
    if p is not None:
        src,dst,sp,dp,proto,total,payload,bc,seq,ack,flags=p
        return '.'.join(map(str,src)),'.'.join(map(str,dst)),sp,dp,proto,payload,seq,ack,flags
    # Scapy handles offload records, IPv6 and extension headers for evidence.
    if link==1 and len(data)>=14 and data[12:14] not in [b'\x08\x00',b'\x86\xdd']:return None
    from scapy.all import conf,IP,IPv6,TCP,UDP
    pkt=conf.l2types[link](data);trans=pkt[TCP] if TCP in pkt else (pkt[UDP] if UDP in pkt else None)
    ip=pkt[IP] if IP in pkt else (pkt[IPv6] if IPv6 in pkt else None)
    if ip is None or trans is None:return None
    return ip.src,ip.dst,int(trans.sport),int(trans.dport),6 if TCP in pkt else 17,bytes(trans.payload),int(trans.seq) if TCP in pkt else 0,int(trans.ack) if TCP in pkt else 0,int(trans.flags) if TCP in pkt else 0

def merge_range(ranges,start,end):
    if end<=start:return
    if not ranges:ranges.append([start,end]);return
    if start>ranges[-1][1]:ranges.append([start,end]);return
    if start>=ranges[-1][0]:ranges[-1][1]=max(ranges[-1][1],end);return
    import bisect
    k=bisect.bisect_left(ranges,[start,-1])
    if k and ranges[k-1][1]>=start:k-=1
    stop=k
    while stop<len(ranges) and ranges[stop][0]<=end:
        start=min(start,ranges[stop][0]);end=max(end,ranges[stop][1]);stop+=1
    ranges[k:stop]=[[start,end]]

def one_file(spec):
    record,muse,saltpath=spec;name=record['file'];stem=Path(name).stem
    done=W/'markers/sibling_files'/f'{name}.json';private=PRIVATE/f'{stem}.pickle.gz'
    if done.exists() and private.exists():return json.loads(done.read_text())
    salt=Path(saltpath).read_bytes();packets=Counter();tcp={};udp=set();reader=Capture(Path(muse)/'data/ISCXVPN2016'/name)
    for t,raw,link in reader:
        packets[hashlib.sha256(raw).digest()]+=1
        p=parsed_packet(raw,link)
        if p is None:continue
        src,dst,sp,dp,proto,payload,seq,ack,flags=p
        ep1=(src,sp);ep2=(dst,dp);key=(min(ep1,ep2),max(ep1,ep2),proto)
        ident=hmac.new(salt,repr(key).encode(),hashlib.sha256).digest()
        if proto==17:udp.add(ident);continue
        direction=int(ep1!=key[0])
        entry=tcp.setdefault(ident,{'directions':[None,None]})
        state=entry['directions'][direction]
        if state is None:
            state={'first':None,'last':None,'intervals':[],'signatures':set(),'retransmissions':0,'out_of_order':0,'max_seq':None,'wrap_ambiguous':False,'syn_seq':set()};entry['directions'][direction]=state
        sig=hashlib.sha256(struct.pack('!IIB',seq,ack,flags)+hashlib.sha256(payload).digest()).digest()
        if sig in state['signatures']:state['retransmissions']+=1;continue
        state['signatures'].add(sig)
        if state['max_seq'] is not None:
            if seq<state['max_seq']:
                if state['max_seq']-seq>2**31:state['wrap_ambiguous']=True
                else:state['out_of_order']+=1
        state['max_seq']=seq if state['max_seq'] is None else max(seq,state['max_seq'])
        span=len(payload)+int(bool(flags&2))+int(bool(flags&1));end=seq+span
        if end>=2**32:state['wrap_ambiguous']=True
        if flags&2 and not flags&16:state['syn_seq'].add(seq)
        packet={'time':t,'seq':seq,'ack':ack,'end':end,'flags':flags,'payload_length':len(payload)}
        if state['first'] is None or t<state['first']['time']:state['first']=packet
        if state['last'] is None or t>=state['last']['time']:state['last']=packet
        merge_range(state['intervals'],seq,end)
    assert reader.sha.hexdigest()==record['input_sha256'],name+' input hash changed'
    reader.close()
    for entry in tcp.values():
        for state in entry['directions']:
            if state is not None:state.pop('signatures');state['syn_seq']=sorted(state['syn_seq'])
    d=np.load(W/'data/v3_files'/f'{stem}.npz');keep=d['keep_clean']
    vectors=np.concatenate([d['seq'][keep],d['iat'][keep],d['raw_masked'][keep],d['stat'][keep]],axis=1).astype(np.float32)
    features=Counter(hashlib.sha256(v.tobytes(order='C')).digest() for v in vectors)
    PRIVATE.mkdir(parents=True,exist_ok=True)
    with gzip.open(private,'wb',compresslevel=1) as f:pickle.dump({'packet_hashes':packets,'tcp':tcp,'udp':udp,'features':features},f,protocol=5)
    summary={'file':name,'packet_count':sum(packets.values()),'tcp_five_tuples':len(tcp),'udp_five_tuples':len(udp),'clean_feature_flows':sum(features.values()),'retransmissions':sum(s['retransmissions'] for c in tcp.values() for s in c['directions'] if s),'out_of_order':sum(s['out_of_order'] for c in tcp.values() for s in c['directions'] if s),'connection_identifiers':'HMAC-SHA256 with local random salt; raw endpoints never stored'}
    dump(done,summary);return summary

@functools.lru_cache(maxsize=2)
def evidence(name):
    with gzip.open(PRIVATE/f'{Path(name).stem}.pickle.gz','rb') as f:return pickle.load(f)

def split_connection(a,b):
    uncertain=0
    for d in [0,1]:
        aa=a['directions'][d];bb=b['directions'][d];reverse=a['directions'][1-d]
        if not aa or not bb:continue
        if aa['wrap_ambiguous'] or bb['wrap_ambiguous'] or len(aa['syn_seq'])>1 or len(bb['syn_seq'])>1 or len(aa['intervals'])!=1:
            uncertain+=1;continue
        first=bb['first'];end=aa['intervals'][0][1]
        if first['flags']&2 and not first['flags']&16:continue
        if first['seq']!=end:continue
        if reverse and not reverse['wrap_ambiguous'] and len(reverse['intervals'])==1:
            if first['flags']&16 and first['ack']==reverse['intervals'][0][1]:return True,uncertain
        else:uncertain+=1
    return False,uncertain

def main():
    registration=json.loads((W/'audit/rules_registration.json').read_text());assert hashlib.sha256((W/'RULES.md').read_bytes()).hexdigest()==registration['sha256']
    inv=json.loads((W/'audit/v3_cleaning_inventory.json').read_text());assert inv['complete']
    records={r['file']:r for r in inv['files']}
    saltpath=W/'data/sibling_salt.bin'
    if not saltpath.exists():saltpath.write_bytes(os.urandom(32));saltpath.chmod(0o600)
    muse=json.loads((W/'config/paths.local.json').read_text())['muse_root']
    failed=[];file_summaries=[]
    with ProcessPoolExecutor(max_workers=12) as pool:
        futures={pool.submit(one_file,(r,muse,str(saltpath))):r for r in records.values()}
        for future in as_completed(futures):
            r=futures[future]
            try:x=future.result()
            except Exception as e:
                dump(W/'audit/retries'/f'siblings_{r["file"]}.json',{'file':r['file'],'retry_count':1,'first_error':str(e),'settings_changed':False})
                try:x=pool.submit(one_file,(r,muse,str(saltpath))).result()
                except Exception as second:failed.append({'file':r['file'],'error':str(second)});continue
            file_summaries.append(x);print('EVIDENCE FILE DONE',r['file'],x['packet_count'],flush=True)
    dump(W/'audit/sibling_file_summaries.json',file_summaries)
    if failed:dump(W/'audit/sibling_failed_files.json',failed);return 2
    suffix=defaultdict(list);byclass=defaultdict(list);pairs=defaultdict(set)
    for name,r in records.items():suffix[r['session']].append(name);byclass[r['class']].append(name)
    for names in suffix.values():
        for a,b in itertools.combinations(sorted(names),2):pairs[(a,b)].add('suffix')
    for names in byclass.values():
        names.sort(key=lambda n:(records[n]['first_packet_timestamp'],n))
        for a,b in zip(names,names[1:]):pairs[tuple(sorted([a,b]))].add('adjacent_first_timestamp')
    rows=[];newedges=[];parent={name:name for name in records}
    def root(a):
        while parent[a]!=a:parent[a]=parent[parent[a]];a=parent[a]
        return a
    def union(a,b):
        x,y=root(a),root(b)
        if x!=y:parent[max(x,y)]=min(x,y)
    for names in suffix.values():
        for name in names[1:]:union(names[0],name)
    for (n1,n2),origin in sorted(pairs.items()):
        r1,r2=records[n1],records[n2]
        a,b=(n1,n2) if (r1['min_packet_timestamp'],n1)<=(r2['min_packet_timestamp'],n2) else (n2,n1)
        ra,rb=records[a],records[b]
        overlap=max(ra['min_packet_timestamp'],rb['min_packet_timestamp'])<=min(ra['max_packet_timestamp'],rb['max_packet_timestamp'])
        gap=0. if overlap else max(0.,rb['min_packet_timestamp']-ra['max_packet_timestamp'])
        relation='overlap' if overlap else ('adjacent' if gap<=60 else 'separate')
        ea,eb=evidence(a),evidence(b);ct=set(ea['tcp'])&set(eb['tcp']);cu=ea['udp']&eb['udp'];split=uncertain=0
        if relation=='adjacent':
            for key in ct:
                yes,u=split_connection(ea['tcp'][key],eb['tcp'][key]);split+=int(yes);uncertain+=u
        shared=sum((ea['packet_hashes']&eb['packet_hashes']).values());na=ra['packet_count'];nb=rb['packet_count']
        frac=shared/min(na,nb) if min(na,nb) else 0.;duplicate=min(na,nb)>=100 and frac>=.90
        samefeat=sum((ea['features']&eb['features']).values());common=len(ct)+len(cu)
        if split or relation=='adjacent' and common:verdict='confirmed'
        elif relation in ['adjacent','overlap'] and common==0:verdict='possible'
        elif relation=='separate' and common==0:verdict='no_evidence'
        else:verdict='shared_connection_insufficient'
        row={'file_a':a,'file_b':b,'class':ra['class'],'candidate_origins':'+'.join(sorted(origin)),'time_relation':relation,'gap_seconds':gap,'common_tcp_five_tuples':len(ct),'split_tcp_connections':split,'uncertain_tcp_boundaries':uncertain,'common_udp_five_tuples':len(cu),'identical_packet_count':shared,'identical_packet_fraction_a':shared/na if na else 0.,'identical_packet_fraction_b':shared/nb if nb else 0.,'identical_packet_fraction_min':frac,'identical_feature_flow_count':samefeat,'verdict':verdict,'suspected_duplicate':duplicate}
        rows.append(row)
        if verdict=='confirmed':
            if ra['session']!=rb['session']:newedges.append(row)
            union(a,b)
        print('PAIR DONE',a,b,verdict,split,len(ct),len(cu),flush=True)
    out=W/'audit/sibling_pairs.csv'
    with out.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else []);writer.writeheader();writer.writerows(rows)
    mapping={name:'evidence_'+hashlib.sha256(root(name).encode()).hexdigest()[:16] for name in records}
    dump(W/'audit/evidence_sessions.json',{'file_to_session':mapping,'new_edges':newedges,'suffix_session_count':len(suffix),'evidence_session_count':len(set(mapping.values()))})
    perfile=defaultdict(list)
    for row in rows:
        for name in [row['file_a'],row['file_b']]:perfile[name].append(row)
    with (W/'audit/publishable_sessions.csv').open('w',newline='') as f:
        fields=['file','class','suffix_session','evidence_session','pair_count','confirmed_pairs','possible_pairs','suspected_duplicate_pairs','split_tcp_connections_sum','common_tcp_five_tuples_sum','common_udp_five_tuples_sum','identical_packets_sum','identical_feature_flows_sum']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for name,r in sorted(records.items()):
            ps=perfile[name]
            writer.writerow({'file':name,'class':r['class'],'suffix_session':r['session'],'evidence_session':mapping[name],'pair_count':len(ps),'confirmed_pairs':sum(p['verdict']=='confirmed' for p in ps),'possible_pairs':sum(p['verdict']=='possible' for p in ps),'suspected_duplicate_pairs':sum(p['suspected_duplicate'] for p in ps),'split_tcp_connections_sum':sum(p['split_tcp_connections'] for p in ps),'common_tcp_five_tuples_sum':sum(p['common_tcp_five_tuples'] for p in ps),'common_udp_five_tuples_sum':sum(p['common_udp_five_tuples'] for p in ps),'identical_packets_sum':sum(p['identical_packet_count'] for p in ps),'identical_feature_flows_sum':sum(p['identical_feature_flow_count'] for p in ps)})
    dump(W/'audit/pcap_file_metadata.json',[{k:r[k] for k in ['file','class','first_packet_timestamp','last_packet_timestamp','min_packet_timestamp','max_packet_timestamp','packet_count','header']} for r in records.values()])
    summary={'file_count':len(records),'pair_count':len(rows),'verdict_counts':dict(Counter(r['verdict'] for r in rows)),'suspected_duplicate_pairs':sum(r['suspected_duplicate'] for r in rows),'new_confirmed_edges':len(newedges),'rules_sha256':registration['sha256'],'address_publication':'No raw address or endpoint included; internal connection joins use salted HMAC-SHA256. Salt excluded from return package.'}
    dump(W/'audit/sibling_summary.json',summary);dump(W/'markers/SIBLINGS_COMPLETED.json',summary);return 0
if __name__=='__main__':sys.exit(main())
