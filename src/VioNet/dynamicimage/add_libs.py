import os
import sys
import os.path

def add_path(path):
    if path not in sys.path:
        print('adding {}'.format(path))
        sys.path.insert(0, path)

dinamicimage_dir = os.path.dirname(__file__)
vionetdir = os.path.dirname(dinamicimage_dir)

add_path(os.path.normpath(vionetdir))

# print(dinamicimage_dir)
# print(vionetdir)