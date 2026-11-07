#!/usr/bin/env python3
"""Append the tier binaries (and any extra files, e.g. cygwin1.dll) to a
launcher, producing one self-extracting file. See tools/launcher.cpp for
the trailer layout.

    python3 tools/pack.py LAUNCHER OUT FILE [FILE ...]
"""
import struct, sys

FNV = 1469598103934665603
def fnv1a(data, h=FNV):
    for b in data:
        h = ((h ^ b) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h

def main(argv):
    if len(argv) < 4:
        sys.stderr.write(__doc__); return 1
    launcher, out, files = argv[1], argv[2], argv[3:]
    base = open(launcher, 'rb').read()
    start = len(base)
    payload = bytearray(); records = []
    for path in files:
        data = open(path, 'rb').read()
        name = path.replace('\\', '/').split('/')[-1].encode()
        if len(name) > 31:
            sys.stderr.write('name too long: %s\n' % path); return 1
        records.append(struct.pack('<32sQQQ', name, len(payload), len(data), fnv1a(data)))
        payload += data
        sys.stderr.write('  %-28s %10d bytes\n' % (name.decode(), len(data)))
    trailer = struct.pack('<16sQII', b'ECLOOP-PACK-v1', start, len(records), 0)
    with open(out, 'wb') as o:
        o.write(base); o.write(payload)
        for r in records: o.write(r)
        o.write(trailer)
    sys.stderr.write('%s: %d bytes\n' % (out, start + len(payload) + 56 * len(records) + 32))
    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv))
