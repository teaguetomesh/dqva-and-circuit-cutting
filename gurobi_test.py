#!/usr/bin/python

# Copyright 2019, Gurobi Optimization, LLC

# This example formulates and solves the following simple QCP model:
#  maximize    x
#  subject to  x + y + z = 1
#              x^2 + y^2 <= z^2 (second-order cone)
#              x^2 <= yz        (rotated second-order cone)
#              x, y, z non-negative

from gurobipy import *

# Create a new model
m = Model("qcp")

# Create variables
x = m.addVar(lb=0.0, ub=1.0, vtype=GRB.BINARY, name='x')
not_x = m.addVar(lb=0.0, ub=1.0, vtype=GRB.BINARY, name='not_x')
m.addConstr(not_x == 1-x)

y = m.addVar(lb=0.0, ub=1.0, vtype=GRB.BINARY, name='y')
not_y = m.addVar(lb=0.0, ub=1.0, vtype=GRB.BINARY, name='not_y')
m.addConstr(not_y == 1-y)

tmp1 = m.addVar(lb=0.0, ub=1.0, vtype=GRB.BINARY, name='x&not_y')
m.addConstr(tmp1 == and_(x, not_y))

tmp2 = m.addVar(lb=0.0, ub=1.0, vtype=GRB.BINARY, name='not_x&y')
m.addConstr(tmp2 == and_(not_x, y))

z = m.addVar(lb=0.0, ub=1.0, vtype=GRB.BINARY, name='xXORy')
m.addConstr(z == or_(tmp1,tmp2))

obj = 1.0*z
m.setObjective(obj, GRB.MAXIMIZE)

m.optimize()

for v in m.getVars():
    print('%s %g' % (v.varName, v.x))

print('Obj: %g' % obj.getValue())