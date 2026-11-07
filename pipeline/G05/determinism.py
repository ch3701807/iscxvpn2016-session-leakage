"""Apply the user mandated v3 determinism settings before model construction."""
import os,random
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
def configure(seed):
    import numpy as np,torch
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True,warn_only=True)
    torch.set_num_threads(2)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
