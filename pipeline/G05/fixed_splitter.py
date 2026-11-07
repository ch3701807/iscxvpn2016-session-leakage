"""User supplied FixedSGKF, used without upgrading scikit-learn 1.1.3."""
import numpy as np
from collections import defaultdict
from sklearn.model_selection import StratifiedGroupKFold as _SGKF
from sklearn.utils import check_random_state

class FixedSGKF(_SGKF):
    """StratifiedGroupKFold with the scikit-learn 1.8 shuffle fix (PR #32540), for use under scikit-learn 1.1.3."""
    def _iter_test_indices(self, X, y, groups):
        rng = check_random_state(self.random_state)
        y = np.asarray(y)
        _, y_inv, y_cnt = np.unique(y, return_inverse=True, return_counts=True)
        n_classes = len(y_cnt)
        _, groups_inv, groups_cnt = np.unique(groups, return_inverse=True, return_counts=True)
        ycg = np.zeros((len(groups_cnt), n_classes))
        for ci, gi in zip(y_inv, groups_inv):
            ycg[gi, ci] += 1
        ycf = np.zeros((self.n_splits, n_classes))
        gpf = defaultdict(set)
        if self.shuffle:
            perm = np.arange(len(groups_cnt))
            rng.shuffle(perm)
            ycg = ycg[perm]
            inv_perm = np.empty_like(perm)
            inv_perm[perm] = np.arange(perm.size)
            groups_inv = inv_perm[groups_inv]
        for gi in np.argsort(-np.std(ycg, axis=1), kind="stable"):
            bf = self._find_best_fold(y_counts_per_fold=ycf, y_cnt=y_cnt, group_y_counts=ycg[gi])
            ycf[bf] += ycg[gi]
            gpf[bf].add(gi)
        for i in range(self.n_splits):
            yield [idx for idx, gi in enumerate(groups_inv) if gi in gpf[i]]

# Rebind only the outer splitter in a private copy of the original function globals.
# The original source module/function is never modified.
import types
import source_g01
folds = types.FunctionType(source_g01.folds.__code__,
    dict(vars(source_g01), StratifiedGroupKFold=FixedSGKF),
    name='folds', argdefs=source_g01.folds.__defaults__, closure=source_g01.folds.__closure__)
