"""An authorization releases only the exact historical USTC stop it names."""
from pathlib import Path
import hashlib
import json

W = Path(__file__).resolve().parents[1]


def stop_is_active():
    stop = W / 'markers/STOPPED.json'
    if not stop.exists():
        return False
    release = W / 'markers/PROTOCOL_STOP_RELEASED.json'
    if not release.exists():
        return True
    r = json.loads(release.read_text())
    s = json.loads(stop.read_text())
    return not (r.get('decision_maker') == 'user' and
                r.get('released_stop_sha256') == hashlib.sha256(stop.read_bytes()).hexdigest() and
                r.get('released_stage') == s.get('stage') == 'USTC_inventory' and
                r.get('authorization') == 'G01 continuation 4')


def verify_frozen_rules():
    for filename, marker in [('RULES.md', 'audit/rules_registration.json'),
                             ('RULES_addendum.md', 'markers/RULES_ADDENDUM_FROZEN.json')]:
        expected = json.loads((W / marker).read_text())['sha256']
        assert hashlib.sha256((W / filename).read_bytes()).hexdigest() == expected, filename
