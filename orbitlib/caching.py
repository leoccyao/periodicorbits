from functools import wraps

def cached_function(func):
    """Memoize calls in a dictionary attached to ``func``."""
    func.cache = {}
    @wraps(func)
    def wrapper(*args, **kwargs):
        key = args, frozenset(kwargs.items())
        if key not in func.cache:
            func.cache[key] = func(*args, **kwargs)

        return func.cache[key]
    return wrapper

def cached_method(func):
    """Memoize a method using the owning object's configurable cache."""
    key = func.__name__

    @wraps(func)
    def wrapper(self: Cached_Methods, *args, _cache_controls: tuple[bool, bool, bool], **kwargs):
        if _cache_controls is None:
            _cache_controls = (True, True, True)
        
        _cache_read, _cache_compute, _cache_write = _cache_controls
        value_exists = False
        stale = False
        value = None

        if _cache_read or _cache_write:
            dict_key = key, args, frozenset(kwargs.items())
            cache_dict = getattr(self, self.cache_attr)

        if _cache_read:
            if dict_key in cache_dict:
                value = cache_dict[dict_key]
                stale = True
                value_exists = True

        if _cache_compute and not value_exists:
            value = func(self, *args,
                _cache_controls=_cache_controls,
                **kwargs,
            )
            value_exists = True
            stale = False
        
        if not value_exists:
            raise KeyError(f'Result of function call {key} not in cache')

        if _cache_write and not stale:
            cache_dict[dict_key] = value
            
        return value
    return wrapper


class Cached_Methods(object):
    """Mixin providing the cache storage expected by :func:`cached_method`."""
    cache_attr = '_cache'

    def __init__(self) -> None:
        setattr(self, self.cache_attr, {})

    def _get_cache(self):
        return getattr(self, self.cache_attr)

    def _cache_contains(self, key):
        return key in self._get_cache()

    def _cache_pop(self, key):
        return self._get_cache().pop(key)
