#!/usr/bin/env bash
# Regenerate every number and table of the paper from the released run files, then compare with expected/.
# Usage: ./reproduce.sh            (uses python3 on PATH)
#        PYTHON=/path/to/python ./reproduce.sh
set -euo pipefail
cd "$(dirname "$0")"
export ARTEFACT_ROOT="$PWD"
PY="${PYTHON:-python3}"
"$PY" -I -c "import numpy, sklearn, sys; print('python', sys.version.split()[0], 'numpy', numpy.__version__, 'scikit-learn', sklearn.__version__)"
rm -rf outputs && mkdir outputs
for s in fixed_split_plan check_g03 check_g04 split_bug_effect check_g05 check_leak check_g06 make_numbers; do
  echo "== analysis/$s.py"
  "$PY" -I "analysis/$s.py" | tail -n 1
done
status=0
for f in expected/*; do
  if ! cmp -s "$f" "outputs/$(basename "$f")"; then echo "DIFFERS: $(basename "$f")"; status=1; fi
done
[ "$status" -eq 0 ] && echo "All $(ls expected | wc -l) outputs are byte-identical to expected/."
exit "$status"
