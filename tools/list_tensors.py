"""List tensors (name, dtype, shape) of the safetensors files in a HF model directory.

  python list_tensors.py <dir> [substring ...]   # only names containing any substring
"""

import glob
import json
import os
import struct
import sys

names = {}
for f in glob.glob(os.path.join(sys.argv[1], "*.safetensors")):
    with open(f, "rb") as h:
        (n,) = struct.unpack("<Q", h.read(8))
        for k, v in json.loads(h.read(n)).items():
            if k != "__metadata__":
                names[k] = (v["dtype"], v["shape"])
print(len(names), "tensors")
for k in sorted(names):
    if len(sys.argv) < 3 or any(s in k for s in sys.argv[2:]):
        print(k, *names[k])
