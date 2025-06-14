import gadjid
from gadjid import example, ancestor_aid, oset_aid, parent_aid, shd, sid
import numpy as np

# help(gadjid)

# example.run_parent_aid()

Gtrue = np.array([[0, 1, 1, 1, 1], [0, 0, 1, 1, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=np.int8)
Gguess = np.array([[0, 0, 1, 1, 1], [1, 0, 1, 1, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=np.int8)
print("Gtrue:\n", type(Gtrue))

print(ancestor_aid(Gtrue, Gguess, edge_direction="from row to column"))
print(sid(Gtrue, Gguess, edge_direction="from row to column"))
print(shd(Gtrue, Gguess))
