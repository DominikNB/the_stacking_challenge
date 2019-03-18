import numpy as np
from pyswarm import pso
import stack_helpers as sh


'Start with data'

# box_identificator (#Blue (0), Red (1), Yellow (2), Green (3), Orange (4)),
# box_height,
# box_weight,
# on_stack_weight,
# number_of_boxes
boxes = np.array([[0, 50, 50, 500, 5], [1, 70, 80, 1000, 10], [2, 100, 30, 2000, 7], [3, 110, 10, 600, 9], [4, 150, 100, 600, 10]])
non_combine = np.array([[1, 3], [4, 1], [0, 2], [3, 0]])


# Generate items_vect
items_vect = []
for i in range(boxes.shape[0]):
    items_vect = np.append(items_vect, np.ones(boxes[i, 4]) * i)

# Create bounds for heuristic optimizer
lb = np.zeros(np.sum(boxes[:, 4]))
lb = np.append(lb, 0)
ub = np.ones(np.sum(boxes[:, 4])) * 10
ub = np.append(ub, np.sum(boxes[:, 4]))

# Search for best stack
sh.stapler.opt = True
arguments = (boxes, items_vect, non_combine)
vars, fopt = pso(sh.stapler, lb, ub, args=arguments, debug=True, phip=10, phig=0.99, omega=0.1, minfunc=1e-64, minstep=1e-64, maxiter=60, swarmsize=5000)

# Return the result
sh.stapler.opt = False
arguments = (boxes, items_vect, non_combine)
result, stack = sh.stapler(vars, arguments)
print(stack)