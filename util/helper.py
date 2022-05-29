import os
import random
import string

import numpy as np


def get_git():
    """
    If the current directory is a git repository, this function extracts the hash code, and current branch

    Returns
    -------
    hash : string
     hash code of current commit

    branch : string
     current branch
    """
    try:
        from subprocess import Popen, PIPE

        gitproc = Popen(['git', 'show-ref'], stdout=PIPE)
        (stdout, stderr) = gitproc.communicate()

        gitproc = Popen(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], stdout=PIPE)
        (branch, stderr) = gitproc.communicate()
        branch = branch.decode('utf8').split('\n')[0]
        for row in stdout.decode('utf8').split('\n'):
            if row.find(branch) != -1:
                hash = row.split()[0]
                break
    except:
        hash = None
        branch = None
    return hash, branch


def id_generator(size=4, chars=string.ascii_uppercase + string.digits):
    """generates a random sequence of character of length ``size``"""

    return ''.join(random.choice(chars) for _ in range(size))


def one_hot(a, n=None):
    if n is None:
        n = a.max() + 1
    b = np.zeros((a.shape[0], n))
    b[np.arange(a.shape[0]), a] = 1
    return b


def one_hot_batch(a, n=None):
    if n is None:
        n = a.max() + 1
    b = (np.arange(n) == a[..., None]).astype(int)
    return b


def get_files_starting_with(path, starts_with):
    import os
    names = []
    for file in os.listdir(path):
        if file.startswith(starts_with):
            names.append(os.path.join(path, file))
    names.sort()
    return names


def get_total_pionts(train):
    s = 0
    for v in train.values():
        for t in v:
            s += t['reward'].shape[0]
    return s


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def multinomial_rvs(n, p):
    """
    Sample from the multinomial distribution with multiple p vectors.

    * n must be a scalar.
    * p must an n-dimensional numpy array, n >= 1.  The last axis of p
      holds the sequence of probabilities for a multinomial distribution.

    The return value has the same shape as p.
    """
    count = np.full(p.shape[:-1], n)
    out = np.zeros(p.shape, dtype=int)
    ps = p.cumsum(axis=-1)
    # Conditional probabilities
    with np.errstate(divide='ignore', invalid='ignore'):
        condp = p / ps
    condp[np.isnan(condp)] = 0.0
    for i in range(p.shape[-1]-1, 0, -1):
        binsample = np.random.binomial(count, condp[..., i])
        out[..., i] = binsample
        count -= binsample
    out[..., 0] = count
    return out
def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def fix_seeds():
    # Seed value
    # Apparently you may use different seed values at each stage
    seed_value = 0

    # 1. Set `PYTHONHASHSEED` environment variable at a fixed value
    import os
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    # 2. Set `python` built-in pseudo-random generator at a fixed value
    import random
    random.seed(seed_value)

    # 3. Set `numpy` pseudo-random generator at a fixed value
    import numpy as np
    np.random.seed(seed_value)

    # 4. Set `tensorflow` pseudo-random generator at a fixed value
    import tensorflow as tf
    tf.random.set_seed(seed_value)
