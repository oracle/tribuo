#
# Copyright (c) 2022, Oracle and/or its affiliates. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

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
