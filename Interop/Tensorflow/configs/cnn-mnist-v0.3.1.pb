
R
inputplaceholderPlaceholder*
dtype0*$
shape:���������
2
ConstConst*
valueB
 *  �B*
dtype0
4
Const_1Const*
valueB
 *  C*
dtype0
,
SubSubinputplaceholderConst*
T0
!
DivDivSubConst_1*
T0
D
Const_2Const*%
valueB"             *
dtype0
[
TruncatedNormalTruncatedNormalConst_2*
seed�`*
T0*
dtype0*
seed2 
4
Const_3Const*
dtype0*
valueB
 *���=
-
MulMulTruncatedNormalConst_3*
T0
d
Variable
VariableV2*
dtype0*
	container *
shape: *
shared_name 
Q
AssignAssignVariableMul*
use_locking(*
T0*
validate_shape(
�
Conv2dConv2DDivVariable*
	dilations
*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingSAME
5
Const_4Const*
valueB: *
dtype0
4
Const_5Const*
valueB
 *    *
dtype0
9
FillFillConst_4Const_5*
T0*

index_type0
Z

Variable_1
VariableV2*
shape: *
shared_name *
dtype0*
	container 
V
Assign_1Assign
Variable_1Fill*
use_locking(*
T0*
validate_shape(
F
BiasAddBiasAddConv2d
Variable_1*
T0*
data_formatNHWC

ReluReluBiasAdd*
T0
D
Const_6Const*%
valueB"            *
dtype0
D
Const_7Const*%
valueB"            *
dtype0
_
MaxPool	MaxPoolV2ReluConst_6Const_7*
paddingSAME*
T0*
data_formatNHWC
D
Const_8Const*%
valueB"          @   *
dtype0
]
TruncatedNormal_1TruncatedNormalConst_8*
seed�`*
T0*
dtype0*
seed2 
4
Const_9Const*
dtype0*
valueB
 *���=
1
Mul_1MulTruncatedNormal_1Const_9*
T0
f

Variable_2
VariableV2*
dtype0*
	container *
shape: @*
shared_name 
W
Assign_2Assign
Variable_2Mul_1*
T0*
validate_shape(*
use_locking(
�
Conv2d_1Conv2DMaxPool
Variable_2*
	dilations
*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingSAME
6
Const_10Const*
valueB:@*
dtype0
5
Const_11Const*
valueB
 *���=*
dtype0
=
Fill_1FillConst_10Const_11*
T0*

index_type0
Z

Variable_3
VariableV2*
shape:@*
shared_name *
dtype0*
	container 
X
Assign_3Assign
Variable_3Fill_1*
use_locking(*
T0*
validate_shape(
J
	BiasAdd_1BiasAddConv2d_1
Variable_3*
data_formatNHWC*
T0
"
Relu_1Relu	BiasAdd_1*
T0
E
Const_12Const*%
valueB"            *
dtype0
E
Const_13Const*%
valueB"            *
dtype0
e
	MaxPool_1	MaxPoolV2Relu_1Const_12Const_13*
paddingSAME*
T0*
data_formatNHWC
?
Const_14Const*
valueB:
���������*
dtype0
7
Const_15Const*
valueB:�*
dtype0
2
Const_16Const*
dtype0*
value	B : 
N
ConcatConcatV2Const_14Const_15Const_16*
N*

Tidx0*
T0
<
ReshapeReshape	MaxPool_1Concat*
T0*
Tshape0
7
Const_17Const*
valueB:�*
dtype0
7
Const_18Const*
valueB:�*
dtype0
2
Const_19Const*
value	B : *
dtype0
P
Concat_1ConcatV2Const_17Const_18Const_19*

Tidx0*
T0*
N
^
TruncatedNormal_2TruncatedNormalConcat_1*
seed�`*
T0*
dtype0*
seed2 
5
Const_20Const*
valueB
 *���=*
dtype0
2
Mul_2MulTruncatedNormal_2Const_20*
T0
`

Variable_4
VariableV2*
shared_name *
dtype0*
	container *
shape:
��
W
Assign_4Assign
Variable_4Mul_2*
validate_shape(*
use_locking(*
T0
7
Const_21Const*
valueB:�*
dtype0
5
Const_22Const*
valueB
 *���=*
dtype0
=
Fill_2FillConst_21Const_22*
T0*

index_type0
[

Variable_5
VariableV2*
shape:�*
shared_name *
dtype0*
	container 
X
Assign_5Assign
Variable_5Fill_2*
validate_shape(*
use_locking(*
T0
T
MatMulMatMulReshape
Variable_4*
transpose_a( *
transpose_b( *
T0
'
AddAddMatMul
Variable_5*
T0

Relu_2ReluAdd*
T0
=
Const_23Const*
valueB"   
   *
dtype0
^
TruncatedNormal_3TruncatedNormalConst_23*
seed�`*
T0*
dtype0*
seed2 
5
Const_24Const*
valueB
 *���=*
dtype0
2
Mul_3MulTruncatedNormal_3Const_24*
T0
_

Variable_6
VariableV2*
dtype0*
	container *
shape:	�
*
shared_name 
W
Assign_6Assign
Variable_6Mul_3*
use_locking(*
T0*
validate_shape(
6
Const_25Const*
dtype0*
valueB:

5
Const_26Const*
valueB
 *���=*
dtype0
=
Fill_3FillConst_25Const_26*
T0*

index_type0
Z

Variable_7
VariableV2*
shape:
*
shared_name *
dtype0*
	container 
X
Assign_7Assign
Variable_7Fill_3*
use_locking(*
T0*
validate_shape(
U
MatMul_1MatMulRelu_2
Variable_6*
T0*
transpose_a( *
transpose_b( 
+
Add_1AddMatMul_1
Variable_7*
T0
b
initNoOp^Assign	^Assign_1	^Assign_2	^Assign_3	^Assign_4	^Assign_5	^Assign_6	^Assign_7 "�