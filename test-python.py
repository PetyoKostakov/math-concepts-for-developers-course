import math

def solve_linear_equation(a, b):
    if a == 0:
        if b == 0:
            return "Infinite solutions"
        else:
            return "No solution"
    else:
        return -b / a


print(solve_linear_equation(2,4))