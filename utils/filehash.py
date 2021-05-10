import hashlib
import sys

def file_as_bytes(file):
    with file:
        return file.read()

def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


n = len(sys.argv)
for i in range(1, n):
    the_file = sys.argv[i]
    the_hash = md5(the_file)
    print(the_file, "->", the_hash)

