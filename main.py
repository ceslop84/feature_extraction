from vbow import vbow
from mpeg7 import mpeg7
from gist import gist

if __name__ == '__main__':
    vbow('Images/labels.txt', "Output", True)
    gist('Images/labels.txt', "Output", True)
    mpeg7('Images/labels.txt', "Output", True)
