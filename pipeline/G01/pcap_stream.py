"""Streaming pcap/pcapng reader; records contain exact captured bytes, never written."""
import struct
import hashlib

class Capture:
    def __init__(self, path):
        self.f = open(path, 'rb')
        self.sha = hashlib.sha256()
        self.header = {'pcapng': False, 'interfaces': [], 'sections': 0}
        self.interfaces = []
        self.current_interface = None
        magic = self.read(4)
        if magic == b'\x0a\x0d\x0d\x0a':
            self.ng = True
            self.header['pcapng'] = True
            self.shb()
        else:
            self.ng = False
            formats = {b'\xd4\xc3\xb2\xa1': ('<', 1e6), b'\xa1\xb2\xc3\xd4': ('>', 1e6), b'\x4d\x3c\xb2\xa1': ('<', 1e9), b'\xa1\xb2\x3c\x4d': ('>', 1e9)}
            if magic not in formats:
                raise ValueError('Unknown pcap magic')
            self.endian, self.resol = formats[magic]
            major, minor, zone, sig, snap, link = struct.unpack(self.endian+'HHIIII', self.read(20))
            self.header.update(linktype=link & 0xffff, snaplen=snap, timestamp_units_per_second=int(self.resol), version=[major, minor])
    def read(self, n):
        data = self.f.read(n)
        self.sha.update(data)
        return data
    def shb(self):
        length = self.read(4)
        bom = self.read(4)
        if bom == b'\x4d\x3c\x2b\x1a': self.endian = '<'
        elif bom == b'\x1a\x2b\x3c\x4d': self.endian = '>'
        else: raise ValueError('Bad pcapng BOM')
        size = struct.unpack(self.endian+'I', length)[0]
        if size < 28 or size > 64*1024*1024: raise ValueError('Invalid SHB size')
        tail = self.read(size-12)
        if len(tail) != size-12 or struct.unpack(self.endian+'I', tail[-4:])[0] != size: raise ValueError('Truncated SHB')
        self.header['sections'] += 1
        self.interfaces = []
    def __iter__(self):
        if not self.ng:
            while True:
                hdr = self.read(16)
                if not hdr: break
                if len(hdr) != 16: raise ValueError('Truncated pcap record header')
                sec, sub, caplen, wirelen = struct.unpack(self.endian+'IIII', hdr)
                if caplen > 64*1024*1024: raise ValueError('Unexpected huge packet')
                data = self.read(caplen)
                if len(data) != caplen: raise ValueError('Truncated captured packet')
                self.current_interface = (0,0)
                yield sec+sub/self.resol, data, self.header['linktype']
            return
        while True:
            blocktype = self.read(4)
            if not blocktype: break
            if len(blocktype) != 4: raise ValueError('Truncated block type')
            if blocktype == b'\x0a\x0d\x0d\x0a':
                self.shb(); continue
            size_bytes = self.read(4)
            if len(size_bytes) != 4: raise ValueError('Truncated block size')
            size = struct.unpack(self.endian+'I', size_bytes)[0]
            if size < 12 or size % 4 or size > 64*1024*1024: raise ValueError('Invalid pcapng block size')
            rest = self.read(size-8)
            if len(rest) != size-8 or struct.unpack(self.endian+'I', rest[-4:])[0] != size: raise ValueError('Truncated pcapng block')
            body = rest[:-4]
            kind = struct.unpack(self.endian+'I', blocktype)[0]
            if kind == 1:
                link, snap = struct.unpack(self.endian+'HxxI', body[:8]); resol=1e6
                pos=8
                while pos+4<=len(body):
                    code, n=struct.unpack(self.endian+'HH', body[pos:pos+4]);pos+=4
                    val=body[pos:pos+n];pos+=(n+3)//4*4
                    if code==0:break
                    if code==9 and len(val)==1:resol=(2 if val[0]&128 else 10)**(val[0]&127)
                self.interfaces.append((link,snap,resol))
                self.header['interfaces'].append({'section':self.header['sections']-1,'interface':len(self.interfaces)-1,'linktype':link,'snaplen':snap,'timestamp_units_per_second':int(resol)})
            elif kind in [2,6]:
                if kind==6: interface,hi,lo,caplen,wirelen=struct.unpack(self.endian+'5I',body[:20])
                else: interface,drops,hi,lo,caplen,wirelen=struct.unpack(self.endian+'HH4I',body[:20])
                if interface>=len(self.interfaces):raise ValueError('Invalid pcapng interface')
                link,snap,resol=self.interfaces[interface]
                data=body[20:20+caplen]
                if len(data)!=caplen:raise ValueError('Truncated pcapng packet')
                self.current_interface = (self.header['sections']-1,interface)
                yield ((hi<<32)+lo)/resol,data,link
            elif kind==3:
                raise ValueError('SPB has no timestamp; explicit handling required')
    def close(self): self.f.close()

def ipv4_transport(data, linktype):
    """Match v2 IPv4 TCP/UDP parsing. Return ephemeral addresses/ports only."""
    broadcast=False
    if linktype==1:
        if len(data)<14:return None
        broadcast=data[:6]==b'\xff'*6
        proto=int.from_bytes(data[12:14],'big');off=14
        while proto in [0x8100,0x88a8,0x9100]:
            if len(data)<off+4:return None
            proto=int.from_bytes(data[off+2:off+4],'big');off+=4
        if proto!=0x0800:return None
    elif linktype in [101,228]:off=0
    elif linktype==113:
        if len(data)<16 or data[14:16]!=b'\x08\x00':return None
        off=16
    elif linktype==276:
        if len(data)<20 or data[:2]!=b'\x08\x00':return None
        off=20
    elif linktype in [0,108]:
        if len(data)<5 or data[4]>>4!=4:return None
        off=4
    else:raise ValueError('Unsupported linktype '+str(linktype))
    if len(data)<off+20 or data[off]>>4!=4:return None
    ihl=(data[off]&15)*4
    if ihl<20 or len(data)<off+ihl:return None
    total=int.from_bytes(data[off+2:off+4],'big')
    # v2 Scapy does not expose transport for noninitial fragments.
    frag=int.from_bytes(data[off+6:off+8],'big')
    if frag&0x1fff:return None
    proto=data[off+9]
    if proto not in [6,17]:return None
    src=tuple(data[off+12:off+16]);dst=tuple(data[off+16:off+20])
    start=off+ihl
    if len(data)<start+4:return None
    sport,dport=struct.unpack('!HH',data[start:start+4])
    end=min(len(data),off+total)
    if proto==6:
        if end<start+20:return None
        hlen=(data[start+12]>>4)*4
        if hlen<20 or start+hlen>end:return None
        seq,ack=struct.unpack('!II',data[start+4:start+12]);flags=data[start+13]
    else:
        if end<start+8:return None
        hlen=8;seq=ack=flags=0
    payload=data[start+hlen:end]
    # Include Ethernet/IP padding like bytes(Scapy TCP/UDP.payload), if present.
    payload+=data[end:]
    return src,dst,sport,dport,proto,total,payload,broadcast,seq,ack,flags
