import os
import sys
import torch
import numpy as np
import pandas as pd


def set_recursion():
    sys.setrecursionlimit(100000)


def fix_seed(seed):
    import numpy as np
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def random_mask(input_ids, tokenizer):
    length = len(input_ids[0])
    # 移除pad cls sep
    input_ids = input_ids[input_ids > 0][1:-1]
    prob = np.random.random(len(input_ids))
    source, target = [], []
    source.append(101)
    target.append(-100)
    for p, ids in zip(prob, input_ids):
        if p < 0.15 * 0.8:
            source.append(tokenizer.mask_token_id)
            target.append(ids)
        elif p < 0.15 * 0.9:
            source.append(ids)
            target.append(ids)
        elif p < 0.15:
            source.append(np.random.choice(tokenizer.vocab_size))
            target.append(ids)
        else:
            source.append(ids)
            target.append(-100)

    source.append(102)
    target.append(-100)
    while len(source) < length:
        source.append(0)
        target.append(-100)
    return source, target
