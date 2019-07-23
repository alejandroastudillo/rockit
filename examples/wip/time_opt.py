# time optimal example for mass-spring-damper system

from ocpx import *
from casadi import sumsqr, vertcat
ocp = OcpMultiStage()

stage = ocp.stage(t0=0,T=ocp.free(1.0)) # T initialised at 1, T>=0
m = 10
c = 0.001
k = 1
p = stage.state()
v = stage.state()
u = stage.control()

stage.set_der(p,v)
stage.set_der(v,(u - c*v - k*p)/m)

stage.subject_to(u<=50)
stage.subject_to(u>=-50)
stage.subject_to(v<=10)
stage.subject_to(v>=-10)
 # Time scaling

stage.add_objective(stage.T) # Minimize time

#stage.set_initial(p, (ts, xs)) # Grid
#stage.set_initial(p, sin(stage.t)) # Function

#stage.subject_to(p-0.1>=0.2) # Obstacle avoidance

stage.subject_to(stage.at_t0(p)==0.5)
stage.subject_to(stage.at_tf(p)==0)


# Pick a solution method
ocp.method(DirectMethod(solver='ipopt'))

# Make it concrete for this stage
stage.method(MultipleShooting(N=20,M=6,intg='rk'))
sol = ocp.solve()
print(ocp.OptimalX())