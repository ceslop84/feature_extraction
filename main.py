from vbow import vbow
from mpeg7 import mpeg7
from gist import gist

if __name__ == '__main__':
    mpeg7('Images/labels.txt', "Output")
    gist('Images/labels.txt', "Output")
    vbow('Images/labels.txt', "Output")
