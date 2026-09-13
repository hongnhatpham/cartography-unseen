"""Read-only Windows display topology for exhibition commissioning."""
import ctypes as c
import json

U = c.c_uint32
class Luid(c.Structure):
    _fields_ = [('low', U), ('high', c.c_int32)]
class Source(c.Structure):
    _fields_ = [('adapter', Luid), ('id', U), ('mode', U), ('status', U)]
class Target(c.Structure):
    _fields_ = [('adapter', Luid), ('id', U), ('mode', U), ('technology', U),
                ('rotation', U), ('scaling', U), ('numerator', U), ('denominator', U),
                ('scanline', U), ('available', c.c_int32), ('status', U)]
class DisplayPath(c.Structure):
    _fields_ = [('source', Source), ('target', Target), ('flags', U)]
class ModeData(c.Union):
    _fields_ = [('bytes', c.c_ubyte * 48), ('alignment', c.c_uint64)]
class Mode(c.Structure):
    _fields_ = [('type', U), ('id', U), ('adapter', Luid), ('data', ModeData)]

def snapshot(database=False):
    api = c.WinDLL('user32')
    flags = 4 if database else 2
    for _ in range(3):
        np, nm, topology = U(), U(), U()
        result = api.GetDisplayConfigBufferSizes(flags, c.byref(np), c.byref(nm))
        if result:
            raise OSError(result, 'GetDisplayConfigBufferSizes')
        paths, modes = (DisplayPath * np.value)(), (Mode * nm.value)()
        result = api.QueryDisplayConfig(flags, c.byref(np), paths, c.byref(nm), modes,
                                       c.byref(topology) if database else None)
        if result == 122:
            continue
        if result:
            raise OSError(result, 'QueryDisplayConfig')
        records = []
        for path in paths[:np.value]:
            source = path.source
            records.append(dict(source=[source.adapter.high, source.adapter.low, source.id],
                                target=path.target.id, active=bool(path.flags & 1),
                                available=bool(path.target.available),
                                refresh=path.target.numerator / (path.target.denominator or 1)))
        sources = {tuple(p['source']) for p in records if p['active']}
        return dict(active_paths=sum(p['active'] for p in records),
                    independent_sources=len(sources), paths=records,
                    database_topology=topology.value if database else None)
    raise RuntimeError('Display topology kept changing while querying')

if __name__ == '__main__':
    result = dict(active=snapshot())
    try:
        result['saved'] = snapshot(True)
    except OSError as exc:
        result['saved_error'] = str(exc)
    print(json.dumps(result, indent=2))
