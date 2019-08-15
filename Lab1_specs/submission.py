## import modules here 

################# Question 0 #################

def add(a, b): # do not change the heading of the function
    return a + b


################# Question 1 #################

def nsqrt(x): # do not change the heading of the function
    # **replace** this line with your code
    for i in range(0,x//2):
        if x >= i * i:
            pass
        else:
            return i-1
            break

################# Question 2 #################


# x_0: initial guess
# EPSILON: stop when abs(x - x_new) < EPSILON
# MAX_ITER: maximum number of iterations

## NOTE: you must use the default values of the above parameters, do not change them

def find_root(f, fprime, x_0=1.0, EPSILON = 1E-7, MAX_ITER = 1000): # do not change the heading of the function
    iteration = 0
    while iteration < MAX_ITER:
        x_0 = x_0 - f(x_0)/fprime(x_0)
        if f(x_0) == 0:
            return x_0
            break
        else:
            iteration += 1


################# Question 3 #################

class Tree(object):
    def __init__(self, name='ROOT', children=None):
        self.name = name
        self.children = []
        if children is not None:
            for child in children:
                self.add_child(child)
    def __repr__(self):
        return self.name
    def add_child(self, node):
        assert isinstance(node, Tree)
        self.children.append(node)

def make_tree(tokens):  # do not change the heading of the function
    level = 0
    father_tree_list = []
    new_tree_list = []
    current_cursor = 0
    count = 0
    for string in tokens:
        if string == "[":
            father_cursor = current_cursor - 1  # previous one is the father.
            father_tree_list.append((Tree(str(tokens[father_cursor])), level))  # generate father tree
            level += 1
        elif string == "]":             # children append end, start adding tree
            level -= 1
        else:                           # found a value
            new_tree_list.append((Tree(str(string)), level))
        current_cursor += 1         # no matter happens, cursor + 1
    for t in new_tree_list:
        count += 1
        for f in new_tree_list[:count][::-1]:
            if f[1] == t[1] - 1:      # if it is the father
                f[0].add_child(t[0])
                break
            else:
                continue 
    return new_tree_list[0][0]

def max_depth(root): # do not change the heading of the function
    depth = 0
    if root:
        depth += 1
        for i in root.children:
            children_depth = max_depth(i)
            if (children_depth + 1) >= depth:
                depth = children_depth + 1
            else:
                continue
    return depth