from scipy.linalg import cholesky
from numpy import ndarray, transpose

def printArrayAsJavaDoubles(nda: ndarray):
    assert len(nda.shape) == 1
    return "new double[]{"+(", ".join(map(str, nda)))+"}"

def printMatrixAsJavaDoubles(nda: ndarray):
    assert len(nda.shape) == 2
    return "new double[][]{"+(", ".join(map(printArrayAsJavaDoubles, nda))) + "}"

'''
This code can be useful to generate expected values for generating expected values for lu factorization. Here are the steps:
1) Generate a DenseMatrix using whatever method/mechanism you choose
2) Print out the DenseMatrix using org.tribuo.math.la.DenseMatrixTest.printMatrixPythonFriendly(DenseMatrix)
3) The above will print out a python-friendly matrix defined as arrays which you can paste into the code below.
4) Run the code below.  It will print out Java-friendly lower factorization matrix.
'''


if __name__ == '__main__':
    a = [[ 8.000000000000000, 2.000000000000000, 3.000000000000000],
[ 2.000000000000000, 9.000000000000000, 3.000000000000000],
[ 3.000000000000000, 3.000000000000000, 6.000000000000000]]
    c = transpose(cholesky(a))
    print(printMatrixAsJavaDoubles(c))
