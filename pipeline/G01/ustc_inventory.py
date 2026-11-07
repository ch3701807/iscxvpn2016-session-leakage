"""USTC-TFC2016 source validation and CPU-only v3 flow inventory; never trains."""
import csv,hashlib,json,os,subprocess,sys,urllib.request,urllib.parse
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor,as_completed
from pcap_stream import Capture
from extract_v3 import ipv4_transport,PORTS,session,dump
from protocol_control import stop_is_active
W=Path(__file__).resolve().parents[1]
EXPECTED=['BitTorrent','Facetime','FTP','Gmail','MySQL','Outlook','Skype','SMB','Weibo','WorldOfWarcraft','Cridex','Geodo','Htbot','Miuref','Neris','Nsis-ay','Shifu','Tinba','Virut','Zeus']
SOURCE='https://github.com/yungshenglu/USTC-TFC2016'
LIMIT=30_000_000_000

def inventory_file(spec):
    path,cname,relative=spec;path=Path(path)
    fid='capture_'+hashlib.sha256(relative.encode()).hexdigest()[:20]
    marker=W/'markers/ustc_files'/f'{fid}.json'
    if marker.exists():return json.loads(marker.read_text())
    before=after=n=0;first=last=lo=hi=None;flows={};reader=Capture(path)
    for t,raw,link in reader:
        n+=1
        if first is None:first=lo=hi=t
        last=t;lo=min(lo,t);hi=max(hi,t)
        p=ipv4_transport(raw,link)
        if p is None:continue
        src,dst,sp,dp,proto,total,payload,bc,seq,ack,flags=p
        ep1=(src,sp);ep2=(dst,dp);key=(min(ep1,ep2),max(ep1,ep2),proto)
        old=flows.get(key)
        if old is None or t-old[0]>120:
            if old is not None and old[1]>=5:before+=1;after+=int(not old[2])
            old=[t,0,False];flows[key]=old
        old[0]=t;old[1]+=1
        old[2]|=sp in PORTS or dp in PORTS or 224<=dst[0]<=239 or dst==(255,255,255,255) or bc
    for old in flows.values():
        if old[1]>=5:before+=1;after+=int(not old[2])
    digest=reader.sha.hexdigest();header=reader.header;reader.close()
    record={'file_id':fid,'class':cname,'suffix_session_id':'session_'+hashlib.sha256((cname+':'+session(path.name)).encode()).hexdigest()[:20],'size_bytes':path.stat().st_size,'sha256':digest,'packet_count':n,'first_packet_timestamp':first,'last_packet_timestamp':last,'min_packet_timestamp':lo,'max_packet_timestamp':hi,'time_span_seconds':hi-lo if n else None,'v3_before_clean_flows':before,'v3_clean_flows':after,'header':header}
    dump(marker,record);return record

def main():
    muse=Path(json.loads((W/'config/paths.local.json').read_text())['muse_root']);src=muse/'data/USTC-TFC2016'
    if not src.exists():raise RuntimeError('Existing USTC directory absent; source manifest download required before inventory')
    commit=subprocess.run(['git','-C',str(src),'rev-parse','HEAD'],capture_output=True,text=True).stdout.strip()
    previous=W/'audit/ustc_summary.json'
    network=json.loads(previous.read_text()).get('source_network_check',{}) if previous.exists() else {}
    if network:
        network={**network,'reused_prior_probe':True,'note':'Prior source-specific probe only; no claim that server networking is unavailable.'}
    else:
        network={'attempted':True,'source':SOURCE,'reachable':False}
        try:
            req=urllib.request.Request('https://raw.githubusercontent.com/yungshenglu/USTC-TFC2016/master/README.md',headers={'User-Agent':'G01-dataset-audit'})
            with urllib.request.urlopen(req,timeout=30) as r:content=r.read(1024*1024)
            network.update(reachable=True,readme_bytes=len(content),readme_sha256=hashlib.sha256(content).hexdigest())
        except Exception as e:network.update(error_type=type(e).__name__)
    specs=[];archives=[]
    for category in ['Benign','Malware']:
        for path in sorted((src/category).rglob('*')):
            if not path.is_file():continue
            rel=path.relative_to(src)
            if path.suffix.lower()=='.7z':archives.append({'file':str(rel),'size_bytes':path.stat().st_size})
            if path.suffix.lower() not in ['.pcap','.pcapng']:continue
            cname=rel.parts[1] if len(rel.parts)>2 else path.stem
            specs.append((str(path),cname,str(rel)))
    missing=sorted(set(EXPECTED)-{s[1] for s in specs})
    if missing:raise RuntimeError('Local USTC missing classes '+','.join(missing)+'; do not silently inventory incomplete data')
    records=[];failed=[]
    with ProcessPoolExecutor(max_workers=12) as pool:
        futures={pool.submit(inventory_file,s):s for s in specs}
        for future in as_completed(futures):
            spec=futures[future]
            try:r=future.result()
            except Exception as first:
                fid=hashlib.sha256(spec[2].encode()).hexdigest()[:20]
                dump(W/'audit/retries'/f'ustc_{fid}.json',{'first_error_type':type(first).__name__,'retry_count':1,'settings_changed':False})
                try:r=pool.submit(inventory_file,spec).result()
                except Exception as second:failed.append({'class':spec[1],'file_id':fid,'error_type':type(second).__name__});continue
            records.append(r);print('USTC FILE DONE',r['class'],r['file_id'],r['v3_clean_flows'],flush=True)
    dump(W/'audit/ustc_file_inventory.json',records)
    (W/'audit/USTC_INPUT_SHA256SUMS').write_text(''.join(r['sha256']+'  '+r['file_id']+'\n' for r in sorted(records,key=lambda x:x['file_id'])))
    summaries=[]
    for cname in EXPECTED:
        rs=[r for r in records if r['class']==cname]
        summaries.append({'class':cname,'removed_fraction':(sum(r['v3_before_clean_flows'] for r in rs)-sum(r['v3_clean_flows'] for r in rs))/sum(r['v3_before_clean_flows'] for r in rs) if sum(r['v3_before_clean_flows'] for r in rs) else 0.,'file_count':len(rs),'suffix_sessions_with_clean_flows':len({r['suffix_session_id'] for r in rs if r['v3_clean_flows']}),'before_clean_flows':sum(r['v3_before_clean_flows'] for r in rs),'clean_flows':sum(r['v3_clean_flows'] for r in rs),'capture_bytes':sum(r['size_bytes'] for r in rs)})
        summaries[-1].update(suffix_sessions_total_files=len({r['suffix_session_id'] for r in rs}),suffix_sessions_with_unclean_flows=len({r['suffix_session_id'] for r in rs if r['v3_before_clean_flows']}),cleaning_rule_conflict=summaries[-1]['removed_fraction']>.5)
    summary={'source':SOURCE,'local_source_git_commit':commit,'source_network_check':network,'downloaded_this_run_bytes':0,'reason_existing_data_reused':'Existing source repository and extracted captures are read-only inputs; avoid downloading identical public files again.','classes':summaries,'archive_files':archives,'capture_file_count':len(records),'capture_bytes':sum(r['size_bytes'] for r in records),'failed_files':failed,'all_20_classes_have_at_least_two_clean_sessions':all(c['suffix_sessions_with_clean_flows']>=2 for c in summaries),'classes_with_at_least_two_clean_sessions':[c['class'] for c in summaries if c['suffix_sessions_with_clean_flows']>=2],'file_names_published_as_sha256_ids':True,'training_performed':False}
    dump(W/'audit/ustc_summary.json',summary)
    if failed:return 2
    excessive=[c for c in summaries if c['removed_fraction']>.5]
    if excessive and not (W/'markers/PROTOCOL_STOP_RELEASED.json').exists():
        dump(W/'markers/STOPPED.json',{'stage':'USTC_inventory','reason':'Class cleaning loss exceeds 50 percent under user global stop rule','classes':excessive})
        return 3
    if stop_is_active():return 3
    dump(W/'markers/USTC_COMPLETED.json',{'file_count':len(records),'classes':len(summaries),'downloaded_bytes':0,'source':SOURCE});return 0
if __name__=='__main__':sys.exit(main())
