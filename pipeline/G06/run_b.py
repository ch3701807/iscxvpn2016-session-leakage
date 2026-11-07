"""Objective paired-frame timestamp measurements; frame bytes never leave memory."""
import argparse,array,bisect,ipaddress,platform,pwd,struct
from common import *
from pcap_stream import Capture

def sanitize(value):
 text=value.decode('utf-8',errors='replace');count=0
 patterns=[r'(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])',r'(?i)(?<![0-9a-f])(?:[0-9a-f]{2}[:-]){5}[0-9a-f]{2}(?![0-9a-f])',r'(?i)(?<![0-9a-f])(?:[0-9a-f]{4}\.){2}[0-9a-f]{4}(?![0-9a-f])',r'[A-Za-z0-9_.+-]+@[A-Za-z0-9_.-]+',r'(?:/home/|/Users/)[^/\s]+',r'(?i)[A-Z]:\\Users\\[^\\\s]+',r'\\\\[^\\\s]+',r'(?i)\b(?:DESKTOP|LAPTOP|WIN|HOST|PC)-[a-z0-9-]+\b',r'(?i)\b(?:hostname|host|username|user)\s*[:=]\s*[^,;\s]+',r'(?i)\b[a-z][a-z0-9_-]*(?:\.[a-z0-9_-]+)*\.(?:com|net|org|local|lan|internal|edu|cn|io|co|uk|de)\b']
 known={platform.node(),pwd.getpwuid(os.getuid()).pw_name}
 for p in [W,G01,G05,Path.home()]:
  if 'home' in p.parts:
   i=p.parts.index('home')
   if i+1<len(p.parts):known.add(p.parts[i+1])
 for token in sorted(known,key=len,reverse=True):
  if len(token)>=3:patterns.append(r'(?<![\w-])'+re.escape(token)+r'(?![\w-])')
 for pattern in patterns:text,n=re.subn(pattern,'<redacted>',text);count+=n
 def ipv6(m):
  nonlocal count
  s=m.group(0)
  try:ipaddress.IPv6Address(s);count+=1;return '<redacted>'
  except ValueError:return s
 text=re.sub(r'(?i)(?<![\w:])[0-9a-f]*:[0-9a-f:]+(?:%[\w]+)?(?![\w:])',ipv6,text)
 return text,count

def options(body,pos,endian):
 out={}
 while pos+4<=len(body):
  code,n=struct.unpack(endian+'HH',body[pos:pos+4]);pos+=4
  if code==0:break
  assert pos+n<=len(body),'Invalid option size'
  out.setdefault(code,[]).append(body[pos:pos+n]);pos+=(n+3)//4*4
 return out

def metadata(path):
 public={'format':None,'sections':[],'interfaces':[],'redaction_count':0};private={'if_name':[],'if_description':[],'shb_hardware':[]}
 with Path(path).open('rb') as f:
  first=f.read(4)
  if first!=b'\x0a\x0d\x0d\x0a':
   formats={b'\xd4\xc3\xb2\xa1':('<',1000000),b'\xa1\xb2\xc3\xd4':('>',1000000),b'\x4d\x3c\xb2\xa1':('<',1000000000),b'\xa1\xb2\x3c\x4d':('>',1000000000)}
   endian,resol=formats[first];major,minor,zone,sig,snap,link=struct.unpack(endian+'HHIIII',f.read(20))
   public.update(format='pcap',linktypes=[link&65535],snaplens=[snap],timestamp_resolutions=[{'units_per_second':resol,'seconds_per_unit':1/resol}],version=[major,minor])
   return public,private
  public['format']='pcapng';f.seek(0);section=-1;interface=0;endian=None
  while True:
   kindraw=f.read(4)
   if not kindraw:break
   assert len(kindraw)==4;length=f.read(4)
   if kindraw==b'\x0a\x0d\x0d\x0a':
    bom=f.read(4);endian='<' if bom==b'\x4d\x3c\x2b\x1a' else '>';assert bom in [b'\x4d\x3c\x2b\x1a',b'\x1a\x2b\x3c\x4d'];size=struct.unpack(endian+'I',length)[0];assert 28<=size<=64*1024*1024
    rest=f.read(size-12);assert len(rest)==size-12 and struct.unpack(endian+'I',rest[-4:])[0]==size
    body=bom+rest[:-4];section+=1;interface=0;opt=options(body,16,endian)
    row={'section':section,'shb_userappl':[],'shb_os':[],'redaction_count':0,'redactions_by_field':{}}
    for code,label in [(4,'shb_userappl'),(3,'shb_os')]:
     field_count=0
     for val in opt.get(code,[]):
      text,count=sanitize(val);row[label].append(text);field_count+=count
     row['redactions_by_field'][label]=field_count;row['redaction_count']+=field_count
    public['redaction_count']+=row['redaction_count'];public['sections'].append(row)
    if 2 in opt:private['shb_hardware'].append((section,opt[2]))
   else:
    assert endian is not None;size=struct.unpack(endian+'I',length)[0];assert 12<=size<=64*1024*1024 and size%4==0;kind=struct.unpack(endian+'I',kindraw)[0]
    if kind==1:
     rest=f.read(size-8);assert len(rest)==size-8 and struct.unpack(endian+'I',rest[-4:])[0]==size;body=rest[:-4];link,snap=struct.unpack(endian+'HxxI',body[:8]);opt=options(body,8,endian)
     raw=opt.get(9,[b'\x06'])[-1];assert len(raw)==1;base=2 if raw[0]&128 else 10;exponent=raw[0]&127;resol=base**exponent
     row={'section':section,'interface':interface,'linktype':link,'snaplen':snap,'timestamp_resolution_base':base,'timestamp_resolution_exponent':exponent,'timestamp_units_per_second':resol,'seconds_per_unit':1/resol,'timestamp_resolution_option_present':9 in opt,'timestamp_offset_seconds':struct.unpack(endian+'q',opt[14][-1])[0] if 14 in opt else 0}
     public['interfaces'].append(row)
     for code,label in [(2,'if_name'),(3,'if_description')]:
      if code in opt:private[label].append((section,interface,opt[code]))
     interface+=1
    else:
     f.seek(size-12,1);assert struct.unpack(endian+'I',f.read(4))[0]==size
  public['linktypes']=sorted(set(r['linktype'] for r in public['interfaces']));public['snaplens']=sorted(set(r['snaplen'] for r in public['interfaces']));public['timestamp_resolutions']=[{'section':r['section'],'interface':r['interface'],'units_per_second':r['timestamp_units_per_second'],'seconds_per_unit':r['seconds_per_unit']} for r in public['interfaces']]
 return public,private

def file_packets(path,expected):
 reader=Capture(path);hashes=bytearray();times=array.array('d');tmin=float('inf');tmax=-float('inf');count=0
 try:
  for ts,frame,link in reader:
   hashes.extend(hashlib.sha256(frame).digest());times.append(ts);tmin=min(tmin,ts);tmax=max(tmax,ts);count+=1
  digest=reader.sha.hexdigest();assert digest==expected,'Raw capture digest differs from G01 inventory'
 finally:reader.close()
 out=np.empty(count,dtype=[('hash','S32'),('time','<f8'),('index','<i8')]);out['hash']=np.frombuffer(hashes,dtype='S32');out['time']=np.frombuffer(times,dtype=np.float64);out['index']=np.arange(count)
 out.sort(order=['hash','time','index'],kind='stable')
 meta,private=metadata(path)
 return out,{'packet_count':count,'min_timestamp':tmin if count else None,'max_timestamp':tmax if count else None,'total_timespan_seconds':tmax-tmin if count else None,'input_sha256':digest,'metadata':meta},private

def inversion_count(values):
 if len(values)<2:return 0
 unique=np.unique(values);ranks=np.searchsorted(unique,values)+1;tree=np.zeros(len(unique)+1,dtype=np.int64);inv=0
 for i,k in enumerate(ranks):
  q=int(k);less_equal=0
  while q:less_equal+=int(tree[q]);q-=q&-q
  inv+=i-less_equal;q=int(k)
  while q<len(tree):tree[q]+=1;q+=q&-q
 return inv

def process_pair(index):
 guard();pairs=read(W/'config/pairs.json');a,b=pairs[index];pairid=f'pair_{index+1:02d}';dest=W/'audit/packet_pairs'/f'{pairid}.json';marker=W/'markers/packet_pairs'/f'{pairid}.json'
 if marker.exists():assert sha(dest)==read(marker)['json_sha256'];return
 inventory={r['file']:r for r in read(G01/'audit/v3_cleaning_inventory.json')['files']}
 with (G01/'audit/sibling_pairs.csv').open(newline='') as f:rows=list(csv.DictReader(f))
 original=next(r for r in rows if {r['file_a'],r['file_b']}=={a,b});expected=int(original['identical_packet_count'])
 rawroot=Path(read(G01/'config/paths.local.json')['muse_root'])/'data/ISCXVPN2016'
 x,ma,pa=file_packets(rawroot/a,inventory[a]['input_sha256']);print(json.dumps({'pair':pairid,'side':'a','packets':len(x)}),flush=True)
 z,mb,pb=file_packets(rawroot/b,inventory[b]['input_sha256']);print(json.dumps({'pair':pairid,'side':'b','packets':len(z)}),flush=True)
 hx,ix,cx=np.unique(x['hash'],return_index=True,return_counts=True);hz,iz,cz=np.unique(z['hash'],return_index=True,return_counts=True)
 common,ia,ib=np.intersect1d(hx,hz,assume_unique=True,return_indices=True);n=int(np.minimum(cx[ia],cz[ib]).sum());matched=np.empty(n,dtype=[('ta','<f8'),('ai','<i8'),('tb','<f8'),('bi','<i8')]);cursor=0
 for p,q in zip(ia,ib):
  m=min(int(cx[p]),int(cz[q]));sl=slice(cursor,cursor+m);matched['ta'][sl]=x['time'][ix[p]:ix[p]+m];matched['tb'][sl]=z['time'][iz[q]:iz[q]+m];matched['ai'][sl]=x['index'][ix[p]:ix[p]+m];matched['bi'][sl]=z['index'][iz[q]:iz[q]+m];cursor+=m
 assert cursor==n
 dt=matched['tb']-matched['ta'];ad=np.abs(dt);counts={'dt_eq_0':int(sum(dt==0)),'abs_dt_le_1us':int(sum(ad<=1e-6)),'abs_dt_le_1ms':int(sum(ad<=1e-3)),'abs_dt_le_1s':int(sum(ad<=1)),'abs_dt_gt_1s':int(sum(ad>1))}
 quant=np.quantile(dt,[0,.05,.5,.95,1]).tolist() if n else [None]*5
 matched.sort(order=['ta','ai','tb','bi'],kind='stable');inversions=inversion_count(matched['tb']);possible=n*(n-1)//2;adj=int(sum(np.diff(matched['tb'])<0));private_summary={k:{'equal':pa[k]==pb[k],'a_option_record_count':len(pa[k]),'b_option_record_count':len(pb[k])} for k in pa}
 for side,meta,t in [('a',ma,matched['ta']),('b',mb,matched['tb'])]:
  span=float(t.max()-t.min()) if n else None;meta.update(matched_packet_count=n,matched_packet_fraction=n/meta['packet_count'] if meta['packet_count'] else None,matched_min_timestamp=float(t.min()) if n else None,matched_max_timestamp=float(t.max()) if n else None,matched_timespan_seconds=span,matched_timespan_fraction=span/meta['total_timespan_seconds'] if n and meta['total_timespan_seconds'] else None)
 result={'pair':pairid,'file_a':a,'file_b':b,'status':'completed','matched_packet_count':n,'original_G01_identical_packet_count':expected,'G01_count_matches':n==expected,'common_hash_count':len(common),'dt_counts':counts,'dt_proportions':{k:v/n if n else None for k,v in counts.items()},'dt_quantiles_seconds':dict(zip(['min','q05','median','q95','max'],quant)),'timestamp_order':{'b_monotonic_nondecreasing':inversions==0,'strict_inversion_count':int(inversions),'possible_ordered_index_pairs':possible,'inversion_pair_fraction':inversions/possible if possible else 0.,'adjacent_descents':adj,'adjacent_descent_fraction':adj/(n-1) if n>1 else 0.,'tied_a_adjacent_count':int(sum(np.diff(matched['ta'])==0))},'a':ma,'b':mb,'private_option_equality_only':private_summary,'redaction_count':ma['metadata']['redaction_count']+mb['metadata']['redaction_count'],'measurement_only_no_causal_interpretation':True,'rules_sha256':sha(W/'RULES_G06.md')}
 dump(dest,result);guard();dump(marker,{'time':now(),'json_sha256':sha(dest),'matched_packet_count':n,'G01_count_matches':n==expected});print(json.dumps({'pair':pairid,'completed':True,'matched_packets':n,'G01_count_matches':n==expected}),flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--pair',type=int,required=True,choices=range(13));a=p.parse_args();process_pair(a.pair)
if __name__=='__main__':main()
