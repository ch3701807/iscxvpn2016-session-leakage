"""B+3: packet headers only; reuse pre-existing flow-level TLS counters."""
import csv
import datetime
import hashlib
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from extract_v3 import dump
from pcap_stream import Capture
from protocol_control import stop_is_active, verify_frozen_rules

W = Path(__file__).resolve().parents[1]
TARGETS = {17: {500: 'UDP_500', 4500: 'UDP_4500', 1194: 'UDP_1194'},
           6: {443: 'TCP_443', 1723: 'TCP_1723'}}
PORT_FIELDS = ['UDP_500', 'UDP_4500', 'UDP_1194', 'TCP_443', 'TCP_1723',
               'any_specified_port', 'other', 'unavailable']


def packet_headers(raw, link):
    """Return version, final protocol, optional ports, fragment flag, status."""
    if link == 1:
        if len(raw) < 14:
            return None, None, None, False, 'malformed_link'
        kind, offset = int.from_bytes(raw[12:14], 'big'), 14
        while kind in [0x8100, 0x88a8, 0x9100]:
            if len(raw) < offset + 4:
                return None, None, None, False, 'malformed_link'
            kind = int.from_bytes(raw[offset + 2:offset + 4], 'big')
            offset += 4
        if kind not in [0x0800, 0x86dd]:
            return None, None, None, False, 'non_ip'
    elif link == 113:
        if len(raw) < 16:
            return None, None, None, False, 'malformed_link'
        if int.from_bytes(raw[14:16], 'big') not in [0x0800, 0x86dd]:
            return None, None, None, False, 'non_ip'
        offset = 16
    elif link == 276:
        if len(raw) < 20:
            return None, None, None, False, 'malformed_link'
        if int.from_bytes(raw[:2], 'big') not in [0x0800, 0x86dd]:
            return None, None, None, False, 'non_ip'
        offset = 20
    elif link in [101, 228, 229]:
        offset = 0
    elif link in [0, 108]:
        offset = 4
    else:
        return None, None, None, False, 'unsupported_link'
    if len(raw) <= offset:
        return None, None, None, False, 'malformed_ip'
    version = raw[offset] >> 4
    fragment = False
    if version == 4:
        if len(raw) < offset + 20:
            return None, None, None, False, 'malformed_ip'
        header_length = (raw[offset] & 15) * 4
        if header_length < 20 or len(raw) < offset + header_length:
            return None, None, None, False, 'malformed_ip'
        protocol = raw[offset + 9]
        fragment_bits = int.from_bytes(raw[offset + 6:offset + 8], 'big')
        fragment = bool(fragment_bits & 0x3fff)
        transport = offset + header_length
        noninitial = bool(fragment_bits & 0x1fff)
    elif version == 6:
        if len(raw) < offset + 40:
            return None, None, None, False, 'malformed_ip'
        protocol, transport = raw[offset + 6], offset + 40
        noninitial = False
        for _ in range(32):
            if protocol in [0, 43, 60, 135, 139, 140]:
                if len(raw) < transport + 2:
                    return 6, protocol, None, fragment, 'truncated_extension'
                next_protocol = raw[transport]
                length = (raw[transport + 1] + 1) * 8
            elif protocol == 51:
                if len(raw) < transport + 2:
                    return 6, protocol, None, fragment, 'truncated_extension'
                next_protocol = raw[transport]
                length = (raw[transport + 1] + 2) * 4
            elif protocol == 44:
                if len(raw) < transport + 8:
                    return 6, protocol, None, True, 'truncated_extension'
                next_protocol = raw[transport]
                bits = int.from_bytes(raw[transport + 2:transport + 4], 'big')
                fragment = True
                noninitial = bool(bits & 0xfff8)
                length = 8
            else:
                break
            if len(raw) < transport + length:
                return 6, protocol, None, fragment, 'truncated_extension'
            transport += length
            protocol = next_protocol
            if noninitial:
                break
        else:
            return 6, protocol, None, fragment, 'extension_limit'
    else:
        return None, None, None, False, 'malformed_ip'
    if protocol not in [6, 17]:
        return version, protocol, None, fragment, 'ip'
    if noninitial:
        return version, protocol, None, fragment, 'noninitial_fragment'
    if len(raw) < transport + 4:
        return version, protocol, None, fragment, 'truncated_transport'
    ports = (int.from_bytes(raw[transport:transport + 2], 'big'),
             int.from_bytes(raw[transport + 2:transport + 4], 'big'))
    return version, protocol, ports, fragment, 'ip'


def ratio(a, b):
    return a / b if b else None


def one_file(spec):
    record, muse = spec
    name = record['file']
    marker = W / 'markers/vpn_metadata_files' / (name + '.json')
    if marker.exists():
        return json.loads(marker.read_text())
    versions, protocols, per_version_protocol, ports = Counter(), Counter(), Counter(), Counter()
    statuses, links, interfaces, fragments = Counter(), Counter(), Counter(), Counter()
    reader = Capture(Path(muse) / 'data/ISCXVPN2016' / name)
    n = 0
    for _, raw, link in reader:
        n += 1
        links[str(link)] += 1
        section, interface = reader.current_interface
        interfaces[(section, interface, link)] += 1
        version, protocol, pair, fragment, status = packet_headers(raw, link)
        statuses[status] += 1
        if version is None:
            continue
        versions[str(version)] += 1
        protocols[str(protocol)] += 1
        per_version_protocol[f'{version}:{protocol}'] += 1
        if fragment:
            fragments[str(version)] += 1
        if protocol in TARGETS:
            if pair is None:
                ports['unavailable'] += 1
            else:
                hits = {TARGETS[protocol][port] for port in pair if port in TARGETS[protocol]}
                if hits:
                    ports['any_specified_port'] += 1
                    ports.update(hits)
                else:
                    ports['other'] += 1
    assert reader.sha.hexdigest() == record['input_sha256'], 'Changed input capture'
    assert n == record['packet_count']
    assert sum(statuses.values()) == n == sum(links.values()) == sum(interfaces.values())
    assert sum(protocols.values()) == sum(versions.values())
    assert sum(protocols.get(str(p), 0) for p in [6, 17]) == ports['any_specified_port'] + ports['other'] + ports['unavailable']
    header = reader.header
    reader.close()
    tls = {}
    for stage, nflows, stats in [('before_clean', record['before_flows'], record['sni_before_clean']),
                                  ('after_clean', record['after_flows'], record['sni_after_clean'])]:
        tls[stage] = {'flow_count': nflows, 'clienthello_flows': stats[0], 'sni_flows': stats[1],
                      'clienthello_flow_fraction': ratio(stats[0], nflows),
                      'sni_flow_fraction': ratio(stats[1], nflows)}
    output = {'file': name, 'class': record['class'], 'vpn_file': name.startswith('vpn_'),
              'packet_count': n, 'input_sha256': record['input_sha256'], 'header': header,
              'ip_version_packet_counts': dict(versions), 'ip_protocol_packet_counts': dict(protocols),
              'version_protocol_packet_counts': dict(per_version_protocol),
              'ESP_50_packets': protocols['50'], 'GRE_47_packets': protocols['47'],
              'port_category_packet_counts': {field: ports[field] for field in PORT_FIELDS},
              'packet_status_counts': dict(statuses), 'fragment_packet_counts': dict(fragments),
              'linktype_packet_counts': dict(links),
              'interface_packet_counts': [{'section': s, 'interface': i, 'linktype': link, 'packets': count}
                                         for (s, i, link), count in sorted(interfaces.items())],
              'tls_flow_statistics': tls, 'tls_counts_source': 'existing v3 extraction metadata',
              'new_payload_inspection': False}
    dump(marker, output)
    return output


def main():
    verify_frozen_rules()
    assert not stop_is_active()
    inventory = json.loads((W / 'audit/v3_cleaning_inventory.json').read_text())
    muse = json.loads((W / 'config/paths.local.json').read_text())['muse_root']
    records, failures = [], []
    with ProcessPoolExecutor(max_workers=12) as pool:
        futures = {pool.submit(one_file, (record, muse)): record['file'] for record in inventory['files']}
        for future in as_completed(futures):
            try:
                result = future.result()
                records.append(result)
                print('VPN METADATA DONE', result['file'], result['packet_count'], flush=True)
            except Exception as error:
                failures.append({'file': futures[future], 'error_type': type(error).__name__, 'error': str(error)})
    if failures:
        dump(W / 'audit/vpn_metadata_failed_files.json', failures)
        return 2
    records.sort(key=lambda r: r['file'])
    dump(W / 'audit/vpn_encapsulation_files.json', records)
    with (W / 'audit/vpn_encapsulation_files.csv').open('w', newline='') as f:
        fields = ['file', 'class', 'vpn_file', 'packet_count', 'linktypes', 'ESP_50', 'GRE_47'] + PORT_FIELDS
        fields += [stage + '_' + metric for stage in ['before_clean', 'after_clean']
                   for metric in ['flow_count', 'clienthello_flows', 'sni_flows', 'clienthello_flow_fraction', 'sni_flow_fraction']]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = {key: record[key] for key in ['file', 'class', 'vpn_file', 'packet_count']}
            row.update(linktypes=','.join(sorted(record['linktype_packet_counts'])),
                       ESP_50=record['ESP_50_packets'], GRE_47=record['GRE_47_packets'])
            row.update(record['port_category_packet_counts'])
            for stage, stats in record['tls_flow_statistics'].items():
                row.update({stage + '_' + metric: value for metric, value in stats.items()})
            writer.writerow(row)
    groups = []
    for vpn in [False, True]:
        rs = [r for r in records if r['vpn_file'] == vpn]
        protocols, ports = Counter(), Counter()
        for r in rs:
            protocols.update(r['ip_protocol_packet_counts'])
            ports.update(r['port_category_packet_counts'])
        tls = {}
        for stage in ['before_clean', 'after_clean']:
            n = sum(r['tls_flow_statistics'][stage]['flow_count'] for r in rs)
            ch = sum(r['tls_flow_statistics'][stage]['clienthello_flows'] for r in rs)
            sni = sum(r['tls_flow_statistics'][stage]['sni_flows'] for r in rs)
            tls[stage] = {'flow_count': n, 'clienthello_flows': ch, 'sni_flows': sni,
                          'clienthello_flow_fraction': ratio(ch, n), 'sni_flow_fraction': ratio(sni, n)}
        groups.append({'vpn_file': vpn, 'files': len(rs), 'packets': sum(r['packet_count'] for r in rs),
                       'ip_protocol_packet_counts': dict(protocols),
                       'ESP_50_packets': protocols['50'], 'GRE_47_packets': protocols['47'],
                       'port_category_packet_counts': {field: ports[field] for field in PORT_FIELDS},
                       'tls_flow_statistics': tls})
    dump(W / 'audit/vpn_encapsulation_summary.json', {
        'post_stage_a_addition': True, 'groups': groups,
        'interpretation_limit': 'Headers and visible ClientHello/SNI describe these captures; capture location or actual VPN encapsulation cannot be established from these counts alone.',
        'specified_port_columns_may_overlap': True, 'new_payload_inspection': False,
    })
    dump(W / 'markers/BPLUS3_COMPLETED.json', {
        'completed_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'post_stage_a_addition': True, 'files': len(records), 'header_only': True,
        'rules_addendum_sha256': json.loads((W / 'markers/RULES_ADDENDUM_FROZEN.json').read_text())['sha256'],
    })
    return 0


if __name__ == '__main__':
    sys.exit(main())
