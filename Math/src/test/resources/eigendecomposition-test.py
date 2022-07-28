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


from numpy import  ndarray
from numpy.linalg import eig

def printArrayAsJavaDoubles(nda: ndarray):
    assert len(nda.shape) == 1
    return "new double[]{"+(", ".join(map(str, nda)))+"}"

def printMatrixAsJavaDoubles(nda: ndarray):
    assert len(nda.shape) == 2
    return "new double[][]{"+(", ".join(map(printArrayAsJavaDoubles, nda))) + "}"

'''
This code can be useful to generate expected values for generating expected values for eigendecomposition. Here are the steps:
1) Generate a DenseMatrix using whatever method/mechanism you choose
2) Print out the DenseMatrix using org.tribuo.math.la.DenseMatrixTest.printMatrixPythonFriendly(DenseMatrix)
3) The above will print out a python-friendly matrix defined as arrays which you can paste into the code below.
4) Run the code below.  It will print out Java-friendly eigenvalues and eigenvectors which you can use in your unit test.
'''


if __name__ == '__main__':
    a = [[272.397395096041460, 28.297103495120155, 11.396656147497406, 12.106296359324405, 20.736399065920747, 15.614978237684149, 18.543740510632610, 5.236590376749971],
         [28.297103495120155, 188.191184447367480, 26.048222455749110, 5.502057905049764,
             25.354185394601018, 13.096921987061375, 17.340008675427296, 11.519052526148316],
         [11.396656147497406, 26.048222455749110, 184.808999941204970, 8.842353725435146,
             15.174247776061687, 12.374742652448266, 13.473116083130513, 10.172399441288023],
         [12.106296359324405, 5.502057905049764, 8.842353725435146, 9.900676311693298,
             12.602839054563070, 14.619263012533160, 21.880922342272704, 4.917793385489778],
         [20.736399065920747, 25.354185394601018, 15.174247776061687, 12.602839054563070,
             185.761164903314860, 17.196093211897440, 19.248597118145410, 9.204408145367047],
         [15.614978237684149, 13.096921987061375, 12.374742652448266, 14.619263012533160,
             17.196093211897440, 152.690212322542980, 15.267218155191648, 10.476862371088615],
         [18.543740510632610, 17.340008675427296, 13.473116083130513, 21.880922342272704,
             19.248597118145410, 15.267218155191648, 448.994124937120900, 12.678295358033086],
         [5.236590376749971, 11.519052526148316, 10.172399441288023, 4.917793385489778, 9.204408145367047, 10.476862371088615, 12.678295358033086, 171.487068866930030]]

    w, vr = eig(a)
    idx = w.argsort()[::-1]
    w = w[idx]
    vr = vr[:, idx]

    print(printArrayAsJavaDoubles(w))
    print(printMatrixAsJavaDoubles(vr))
    