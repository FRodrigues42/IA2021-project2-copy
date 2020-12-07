import testdecisiontrees
from solutions import al029
import datasetstreelearning

def test():
    D,Y,nl,ol = datasetstreelearning.dataset(22)
    return D,Y,nl,ol, al029.createdecisiontree(D, Y)
