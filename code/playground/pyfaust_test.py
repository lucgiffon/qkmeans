from pyfaust.fact import palm4msa
from pyfaust.factparams import ParamsPalm4MSA, ConstraintList, StoppingCriterion
import numpy as np
import pyfaust.proj
# M = np.random.rand(500, 500)
# cons = ConstraintList('splin', 5, 500, 500, 'normcol', 1.0, 500, 500)
#
# # or alternatively using pyfaust.proj
# # from pyfaust.proj import splin, normcol
# # cons = [ splin((500,32), 5), normcol((32,32), 1.0)]
# stop_crit = StoppingCriterion(num_its=200)
# param = ParamsPalm4MSA(cons, stop_crit)
# F = palm4msa(M, param)
# print(F)
from pyfaust import wht
from pyfaust.fact import hierarchical
from pyfaust.factparams import ParamsHierarchical, ConstraintList, StoppingCriterion
from pyfaust.proj import splincol
import numpy as np
import matplotlib.pyplot as plt
#
from pyfaust import wht
from pyfaust.fact import hierarchical
from pyfaust.factparams import ParamsHierarchical, ConstraintList, StoppingCriterion
from pyfaust.proj import splincol
import numpy as np
import matplotlib.pyplot as plt
#
dim = 32
FH = wht(dim)
H = FH.toarray()
nb_fac = int(np.log2(dim))
nb_val_res = [2**(nb_fac-(i)) for i in range(nb_fac)]  # le nombre de valeur par ligne et par colonne dans les residus, je ne comprend pas très bien comment interpréter les résidus
lst_constraints = [splincol((dim, dim), 2).constraint for _ in range(nb_fac)]
fact_cons = ConstraintList(*lst_constraints)
lst_constraints_res = [splincol((dim, dim), i).constraint for i in nb_val_res]

res_cons = ConstraintList(*lst_constraints_res)
stop_crit1 = StoppingCriterion(num_its=200)
stop_crit2 = StoppingCriterion(num_its=200)
param = ParamsHierarchical(fact_cons, res_cons, stop_crit1, stop_crit2,
                           is_update_way_R2L=True)
F = hierarchical(H, param)
F.imshow()
plt.show()
FH.imshow()
plt.show()
print("err=", (FH-F).norm()/FH.norm())

from pyfaust import wht
from pyfaust.fact import hierarchical
from numpy.linalg import norm
# generate a Hadamard Faust of size 32x32
FH = wht(32)
H = FH.toarray() # the full matrix version
# factorize it
FH2 = hierarchical(H, 'squaremat')
FH2.imshow()
plt.show()
# test the relative error
(FH-FH2).norm('fro')/FH.norm('fro') # the result is 1.1015e-16, the factorization is accurate
