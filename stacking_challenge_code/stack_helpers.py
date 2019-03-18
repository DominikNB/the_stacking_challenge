import numpy as np

def stapler(vars, *args):
    if stapler.opt is True:
        boxes = args[0]
        items_vect = args[1]
        non_combine = args[2]
    else:
        boxes = args[0][0]
        items_vect = args[0][1]
        non_combine = args[0][2]

    stack_plan = vars[0:items_vect.shape[0]]
    stack_plan = np.argsort(stack_plan)
    stack = items_vect[stack_plan]
    number_of_boxes = int(round(vars[items_vect.shape[0]]))
    if number_of_boxes == 0:
        number_of_boxes = 1

    stack = stack[0:number_of_boxes]
    if len(stack) > 1:
        stack = sort_the_stack(stack, non_combine)

    stack_int = stack.astype(int)

    # Check for overweight
    weight_on_boxes = boxes[stack_int, 2]
    weight_on_boxes = weight_on_boxes[1:]
    weight_on_boxes = np.flip(weight_on_boxes)
    weight_on_boxes = np.cumsum(weight_on_boxes)
    weight_on_boxes = np.flip(weight_on_boxes)

    if len(weight_on_boxes) != 0:
        allowed_weights = boxes[stack_int, 3]
        allowed_weights = allowed_weights[0:-1]
        perc_weight = weight_on_boxes / allowed_weights
        too_heavy = perc_weight > 1
        weight_tester = np.mean(perc_weight[too_heavy])
        if np.isnan(weight_tester):
            weight_tester = 0
    else:
        weight_tester = 0

    # Check for percentage amount of red boxes
    color_tester = 1-(len(stack_int[stack_int == 1])/len(stack_int))
    if color_tester < 0.9:
        color_tester = 0

    height = np.sum(boxes[stack_int, 1])

    if stapler.opt is True:
        result_constraints = weight_tester + color_tester
        if result_constraints > 0:
            result = result_constraints
        else:
            result = -height
        return result

    else:
        result_constraints = weight_tester
        if result_constraints >= 1:
            result = result_constraints
        else:
            result = -height
        return result, stack_int


def sort_the_stack(test_stack, non_combine):
    forbidden_combination_in_stack = True
    if test_stack.shape[0] > 1:
        while forbidden_combination_in_stack is True:
            forbidden_combination_in_stack = False
            for j in range(non_combine.shape[0]):
                forbidden = non_combine[j]
                values = np.array(test_stack)
                searchval = forbidden
                # Where are forbidden orders? Permute until order is o.k.
                #############################
                N = len(searchval)
                possibles = np.where(values == searchval[0])[0]
                solns = []
                for p in possibles:
                    check = values[p:p + N]
                    if np.all(check == searchval):
                        solns.append(p)
                solns = np.array(solns)
                if solns.shape[0] > 0:
                    forbidden_combination_in_stack = True
                    for g in range(0, solns.shape[0]):
                        a = test_stack[solns[g]]
                        b = test_stack[solns[g] + 1]
                        test_stack[solns[g]] = b
                        test_stack[solns[g] + 1] = a
        sorted_stack = test_stack

        return sorted_stack