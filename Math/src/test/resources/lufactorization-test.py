import numpy as np
from scipy.linalg import lu_factor
from numpy import ndarray

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
4) Run the code below.  It will print out Java-friendly Upper and Lower matrices.
'''


if __name__ == '__main__':
    a = [[ 0.727563680032868, 0.683223471759845, 0.308719455332660, 0.277078490074137, 0.665548951794574, 0.903372264672178, 0.368782913411306, 0.275748069441702, 0.463653575809153, 0.782901778790036, 0.919327782868717, 0.436490974423287, 0.749906181255448, 0.386566874359349, 0.177378477909378, 0.594349910889684, 0.209767568866332, 0.825965871887821, 0.172217937687852, 0.587427381786296],
[ 0.751280406767460, 0.571040348414867, 0.580024884502061, 0.752509948590651, 0.031418238826581, 0.357919919477129, 0.817796930835639, 0.417687546752919, 0.974035681495881, 0.713406257823229, 0.480574516556434, 0.291656497411804, 0.949860134659467, 0.820491823386347, 0.636644547856282, 0.369121493941897, 0.360254875366135, 0.434661085140610, 0.457317094444769, 0.472688420875855],
[ 0.469022520615569, 0.827322424014995, 0.151031554528758, 0.833866235444166, 0.460306372661161, 0.280571991672810, 0.195964207423156, 0.179273440874917, 0.865686796327352, 0.486590661826194, 0.420971706631029, 0.632869791353279, 0.699858645067172, 0.315328450057673, 0.571620305529977, 0.371012289683761, 0.871814595964839, 0.805730942661998, 0.616213685035179, 0.374359356089004],
[ 0.697248729269730, 0.908614580207571, 0.196147071881852, 0.809124816727739, 0.627933275413473, 0.463399271028345, 0.305579155667449, 0.539909434259369, 0.635111014456388, 0.126257823298765, 0.497326892475921, 0.027314166285966, 0.036484516690250, 0.483843854954305, 0.655053381109821, 0.395766280171242, 0.940172465685381, 0.384610843917291, 0.646231978797643, 0.770465637773941],
[ 0.894042795818409, 0.598837037145018, 0.976034471618408, 0.077085112935252, 0.775120695927176, 0.278822302498768, 0.899205329729558, 0.373836143620542, 0.431309555135461, 0.332324278384749, 0.315147471367318, 0.442735942086220, 0.096450915880824, 0.745153306215386, 0.170859737882898, 0.805390719921382, 0.139789595286861, 0.092946816941456, 0.027029866882133, 0.548334691731752],
[ 0.282198286130648, 0.967495322008511, 0.904501079001562, 0.343942146444473, 0.014055818452733, 0.809594124810063, 0.521494303689756, 0.835155555007701, 0.583461648591119, 0.688246069347515, 0.556382201319052, 0.115750123719626, 0.588508636894290, 0.895142285356935, 0.626475165203972, 0.382580991181686, 0.903625094750243, 0.833699601096352, 0.260193631166134, 0.962236220249723],
[ 0.397536156015603, 0.573534718676561, 0.258165001039166, 0.821745557953679, 0.659183474319186, 0.614106754792385, 0.488197374456925, 0.361470612892181, 0.737981632291887, 0.855361490981633, 0.339747108594258, 0.758029864634980, 0.726078204940908, 0.508562234873031, 0.775409918592465, 0.376087287934175, 0.862824158797762, 0.697093313962580, 0.160130754288181, 0.800060876007507],
[ 0.725449553620742, 0.794874724221567, 0.823240915575137, 0.850283450238451, 0.984979205346519, 0.129126686504577, 0.614094824198923, 0.542874681181350, 0.152711743081139, 0.534576009896194, 0.548993732829397, 0.328559964445490, 0.720124675401226, 0.387796158038202, 0.015246621455439, 0.102898367034462, 0.103856897241301, 0.173225648073560, 0.897633823530948, 0.418858419146666],
[ 0.847459486748288, 0.336510352176303, 0.423201682090854, 0.052406745473978, 0.602894469596566, 0.166167160521423, 0.482231721337996, 0.572751749772856, 0.517300271039096, 0.711167603253355, 0.676391229848706, 0.851110551751301, 0.784331937730285, 0.289892235477594, 0.369486380358754, 0.505344107970708, 0.284422759606104, 0.975007583649560, 0.571903178903554, 0.122532257951209],
[ 0.434603504590581, 0.549169111514354, 0.576735419379642, 0.727536503429085, 0.596054173365293, 0.818055527024611, 0.883763059118550, 0.372825589895905, 0.646352552511846, 0.840091222384709, 0.096858386069135, 0.913314494523744, 0.613422506208544, 0.586314598328913, 0.161061316693670, 0.999695346069685, 0.589474508656968, 0.779202691211620, 0.517479473521841, 0.074713999257383],
[ 0.495769399677456, 0.175863201981718, 0.497332761330185, 0.689073450342942, 0.652826173333953, 0.358856625356691, 0.550875603692069, 0.767121644424121, 0.525732565377378, 0.177013501790267, 0.712595067250474, 0.547207516726220, 0.460507804784482, 0.338962509756818, 0.040996406939037, 0.797565866036582, 0.705004072943589, 0.303686967973348, 0.639451838428449, 0.565818461430073],
[ 0.516624213063747, 0.535279223759976, 0.264002422014894, 0.158654828767362, 0.699170735689853, 0.393163530337159, 0.454667182622594, 0.267983090284141, 0.482431089553878, 0.414409509862082, 0.588727323011403, 0.160817772323823, 0.260643008608848, 0.511360607452316, 0.247617535534881, 0.632397980724370, 0.339772817279730, 0.655652397024369, 0.813470907378378, 0.985505725848116],
[ 0.508578384985525, 0.948798121493297, 0.267027195399829, 0.643681174997653, 0.944718115229204, 0.589869906162092, 0.745528957003539, 0.132007862985256, 0.694705669040006, 0.142405761138598, 0.537393927725167, 0.066367400295788, 0.725299650088167, 0.675524011492800, 0.839865909254752, 0.480540012405865, 0.333956520643007, 0.808328728852761, 0.786510616200610, 0.290253712447647],
[ 0.527263841932272, 0.422040341797175, 0.963337127977142, 0.944261290952574, 0.828810195034958, 0.820888510620017, 0.875489401452617, 0.344751271430859, 0.861737213352584, 0.420654751742865, 0.600866554883861, 0.932875128327571, 0.966258030054034, 0.622854420162981, 0.172694559333108, 0.429899839858112, 0.058715605414863, 0.392899165367936, 0.069986240760362, 0.100471268888463],
[ 0.916293775167590, 0.202182492969044, 0.562990240045040, 0.561700943667366, 0.080282170345142, 0.416882590906358, 0.560143976505208, 0.100264341467102, 0.610836098745395, 0.920378070753754, 0.033709461353870, 0.179426442633273, 0.997460814518753, 0.741524133735247, 0.063185128546488, 0.318886141572087, 0.631989300813935, 0.727637943878689, 0.028750514440684, 0.812558114652981],
[ 0.384611053799500, 0.848893556448996, 0.511415204130695, 0.812886941218117, 0.988548992413256, 0.549928653123423, 0.012207078788491, 0.155015083379985, 0.135270634431162, 0.674816309697567, 0.689649109462087, 0.920102844083035, 0.409658210002090, 0.883157153605221, 0.932938256811512, 0.144324939028378, 0.453352172100663, 0.147898709453929, 0.737653484628361, 0.268544896116506],
[ 0.034521076210014, 0.879589794045556, 0.531484787366069, 0.893395046645043, 0.023600080418727, 0.457130087490391, 0.309388398665635, 0.155466514815749, 0.627079557338775, 0.461676268662487, 0.680737820216252, 0.572540137541269, 0.615213870073979, 0.708217434266601, 0.705454587532873, 0.519265491184666, 0.694559974426116, 0.248434532140912, 0.201766115712206, 0.589422611126140],
[ 0.622544320874315, 0.878809666482779, 0.545152555601672, 0.976876449588937, 0.138795391968933, 0.329553074189390, 0.846208536041941, 0.664561531074459, 0.296768764393705, 0.232766896939920, 0.794559626573988, 0.649310518712321, 0.319230116436018, 0.172366521783839, 0.386637431544191, 0.650066810126769, 0.672451256731913, 0.603631652696010, 0.217031435797061, 0.747998180783144],
[ 0.656512480821271, 0.196861300882893, 0.006299617079889, 0.189254768619929, 0.442894631613962, 0.207457828941940, 0.562533957338732, 0.144591463680319, 0.155582437940280, 0.803867184154219, 0.429641056039816, 0.564801166165025, 0.288190176083906, 0.895153914804218, 0.062212252318568, 0.396881986980274, 0.477159937555487, 0.891591046714639, 0.485021313686658, 0.333088241748358],
[ 0.409871211214595, 0.004496524367330, 0.044602411837619, 0.953491004202395, 0.436540393965985, 0.145649394030820, 0.041024471103732, 0.592401749913592, 0.276077042794300, 0.221970716968770, 0.739892799045843, 0.509987512730102, 0.058603584225627, 0.637755395387549, 0.507772241073650, 0.982634356937440, 0.831133393048532, 0.114593073004391, 0.588640743315033, 0.739090953644765]]

    lu, piv = lu_factor(a)
    L, U = np.tril(lu, k=-1) + np.eye(20), np.triu(lu)
    print(printMatrixAsJavaDoubles(U))
    print(printMatrixAsJavaDoubles(L))
