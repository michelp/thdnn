from time import time
from functools import wraps

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print(f"func:{f.__module__}.{f.__name__} took: {te - ts:.4f}")
        return result

    return wrap
