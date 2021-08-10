import os
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = os.path.dirname(__file__)
lib_path = os.path.join(this_dir, '..')

# print('this_dir: ', this_dir)
# print('lib_path: ', lib_path)

add_path(os.path.normpath(lib_path))