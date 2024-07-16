#
#     This file is part of rockit.
#
#     rockit -- Rapid Optimal Control Kit
#     Copyright (C) 2019 MECO, KU Leuven. All rights reserved.
#
#     Rockit is free software; you can redistribute it and/or
#     modify it under the terms of the GNU Lesser General Public
#     License as published by the Free Software Foundation; either
#     version 3 of the License, or (at your option) any later version.
#
#     Rockit is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#     Lesser General Public License for more details.
#
#     You should have received a copy of the GNU Lesser General Public
#     License along with CasADi; if not, write to the Free Software
#     Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
#
#

from ...freetime import FreeTime
from ...casadi_helpers import prepare_build_dir
from ..method import ExternalMethod, linear_coeffs, check_Js
from ...solution import OcpSolution
from ...sampling_method import SamplingMethod

import numpy as np
from casadi import external, vec, CodeGenerator, SX, Sparsity, MX, vcat, veccat, hcat, symvar, substitute, densify, sparsify, DM, Opti, is_linear, vertcat, depends_on, jacobian, linear_coeff, quadratic_coeff, mtimes, pinv, evalf, Function, vvcat, inf, sum1, sum2, diag, solve, fmin, fmax
import casadi
import casadi as cs
from ...casadi_helpers import DM2numpy, reshape_number
from collections import OrderedDict

import subprocess
import os
from ctypes import *
import glob
import shutil
import platform
import hashlib
from casadi import Callback

windows_generator = None
if platform.system()=="Windows":
    if "VCVARS" in os.environ:
        vcvars = os.environ["VCVARS"]
    else:
        vcvars = None
        def vcvars_dirs():
            for p in ["Program Files (x86)","Program Files"]:
                for suffix in ["","Community\\","Professional\\"]:
                    yield f"C:\\{p}\\Microsoft Visual Studio\\2022\\{suffix}VC\Auxiliary\\Build\\vcvars64.bat"
                    yield f"C:\\{p}\\Microsoft Visual Studio\\2019\\{suffix}VC\Auxiliary\\Build\\vcvars64.bat"
                    yield f"C:\\{p}\\Microsoft Visual Studio\\2017\\{suffix}VC\Auxiliary\\Build\\vcvars64.bat"
        for e in vcvars_dirs():
            if os.path.exists(e):
                vcvars = e
                break
        if vcvars is None:
            raise Exception("Could not find vcvars bat file. Tried %s." % str(list(vcvars_dirs())))

        if "2022" in vcvars:
            windows_generator = "Visual Studio 17 2022"
        if "2019" in vcvars:
            windows_generator = "Visual Studio 16 2019"
        if "2017" in vcvars:
            windows_generator = "Visual Studio 15 2017"

def get_terms(e):
    def get_terms_internal(e):
        if e.op()==cs.OP_ADD:
            for i in range(e.n_dep()):
                for t in get_terms_internal(e.dep(i)):
                    yield t
        else:
            yield e
    return list(get_terms_internal(e))


def visit(e,parents=None):
    if parents is None:
        parents = []
    yield (e,parents)
    for i in range(e.n_dep()):
        for t in visit(e.dep(i),parents=[e]+parents):
            yield t

"""
conda install -c conda-forge make cmake
"""

external_dir = os.path.join(os.path.dirname(__file__),"external")

#if not os.path.exists(external_dir):
#    raise Exception("FATROP source not found")

# def run_build(cwd=None):
#     subprocess.run(["bash","build.sh"],cwd=cwd)

INF = inf



stats_fields = [('stop_crit',c_int),('conv_grad',c_int),('conv_con',c_int),('n_outer_iter',c_int),('n_inner_iter',c_int),('runtime',c_double)]

class StatsStruct(Structure):
    _fields_ = stats_fields


def format_float(e):
    return "%0.18f" % e

def strlist(a):
    elems = []
    for e in a:
        if isinstance(e,str):
            elems.append('"'+e+'"')
        elif isinstance(e,float):
            elems.append(format_float(e))
        else:
            elems.append(str(e))
    return ",".join(elems)
    
def check_Js(J):
    """
    Checks if J, a pre-multiplier for slacks, is of legitimate structure
    Empty rows are allowed
    """
    try:
        J = evalf(J)
    except:
        raise Exception("Slack error")
    assert np.all(np.array(J.nonzeros())==1), "All nonzeros must be 1"
    # Check if slice of permutation of unit matrix
    assert np.all(np.array(sum2(J))<=1), "Each constraint can only depend on one slack at most"
    assert np.all(np.array(sum1(J))<=1), "Each constraint must depend on a unique slack, if any"

def fill_in(var, expr,value, array, k):
    value = evalf(value)
    J, r = linear_coeffs(var,expr)
    J = evalf(J)
    r = evalf(r)
    assert r.is_zero()
    check_Js(J)
    expr = reshape_number(var, expr)
    if J.sparsity().get_col():
        array[J.sparsity().get_col(), k] = value[J.row()]
def fill_in_array(var, expr, value, array, numeric = True):
    if numeric:
        value = evalf(value)
    J, r = linear_coeffs(var[:,0],expr)
    J = evalf(J)
    r = evalf(r)
    assert r.is_zero()
    check_Js(J)
    expr = reshape_number(var, expr)
    if J.sparsity().get_col():
        array[J.sparsity().get_col(), :] = value[J.row(), :]
def get_offsets(var, expr):
    J, r = linear_coeffs(var,expr)
    J = evalf(J)
    r = evalf(r)
    assert r.is_zero()
    check_Js(J)
    expr = reshape_number(var, expr)
    return (J.row(), J.sparsity().get_col())

def mark(slack, Js):
    assert np.all(np.array(slack[Js.sparsity().get_col()])==0)
    slack[Js.sparsity().get_col()] = 1

def export_expr(m):
    if isinstance(m,list):
        if len(m)==0:
            return MX(0, 1)
        else:
            return vcat(m)
    return m

def export_num(m):
    res=np.array(evalf(export_expr(m)))
    if np.any(res==-inf) or np.any(res==inf):
        print("WARNING: Double-sided constraints are much preferred. Replaced inf with %f." % INF)
    res[res==-inf] = -INF
    res[res==inf] = INF
    return res

def export_num_vec(m):
    return np.array(evalf(export_expr(m))).reshape(-1)

def export(m):
    return (export_expr(m),False)

def export_vec(m):
    return (export_expr(m),True)


class SourceArtifact:
    def __init__(self, name):
        self.name = name

class LibraryArtifact:
    def __init__(self, basename, relpath):
        self.basename = basename # No starting lib, ending so/dll
        self.relpath = relpath

class FatropMethod(ExternalMethod):
    def __init__(self,
        verbose=False,
        fatrop_options=None,
        intg= 'rk',
        M = 1,
        mode = "fatropy",
        **kwargs):
        self.build_dir_abs = "./foobar"
        supported = {}
        ExternalMethod.__init__(self, supported=supported, **kwargs)
        self.fatrop_options = {} if fatrop_options is None else fatrop_options
        our_defaults = {}
        for k,v in our_defaults.items():
            if k not in self.fatrop_options:
                self.fatrop_options[k] = v
        self.codegen_name = 'casadi_codegen'
        self.user = "((cs_struct*) userparam)"
        self.user_fatrop = "((cs_struct*) fatrop->userparam)"
        self.Nhor = self.N+1
        self.intg = intg
        self.M = M #Assumption!
        # self.N = N
        self.verbose = verbose
        self.artifacts = []
        self.poly_coeff = []  # Optional list to save the coefficients for a polynomial
        self.poly_coeff_z = []  # Optional list to save the coefficients for a polynomial
        self.xk = []  # List for intermediate integrator states
        self.zk = []
        self.samplers = {}
        self.variable_names = {}
        self.mode = mode 
        self.expand = True
    def set_name(self, name):
        self.build_dir_abs = "./"+name 
    def set_expand(self, expand):
        self.expand = expand
    def regname(self, symbol_in, name):
        self.variable_names[symbol_in] = name

    def fill_placeholders_integral(self, phase, stage, expr, *args):
        raise Exception("ocp.integral not supported. Use ocp.sum instead.")
        # if phase==1:
        #     return expr

    def fill_placeholders_sum_control(self, phase, stage, expr, *args):
        if phase == 1:
            return expr
        # raise Exception("ocp.sum not supported. Use ocp.integral instead.")

    def fill_placeholders_sum_control_plus(self, phase, stage, expr, *args):
        if phase == 1:
            return expr

    def fill_placeholders_DT_discrete(self, phase, stage, expr, *args):
        if phase == 1:
            if not stage._state_next:
                raise Exception("Discrete time integrator DT found but dynamics ode is continuous time") 
        return None

    def _register(self,fun_name,argtypes,restype):
        self.prefix=""
        fun = getattr(self.lib,self.prefix+fun_name)
        setattr(self,"_"+fun_name,fun)
        fun.argtypes = argtypes
        fun.restype = restype

    def transcribe_phase1(self, stage, **kwargs):
        # It is not really transcription because FATROP simply takes stage-wise costs as input. Not a large NLP as input.
        #Phase 1 deals with creating placeholder variables and objectives

        self.preamble = ["casadi_int sz_arg=0, sz_res=0, sz_iw=0, sz_w=0;",
                         "casadi_int sz_arg_local, sz_res_local, sz_iw_local, sz_w_local;",
                        ]
        self.postamble = []

        self.stage = stage
        self.opti = Opti()

        # Is computing the whole grid needed for FATROP?
        ## self.time_grid = self.grid(stage._t0, stage._T, self.N)
        self.normalized_time_grid = self.grid(0.0, 1.0, self.N)
        self.time_grid = self.normalized_time_grid
        if self.t_state:
            if isinstance(stage._T, FreeTime):
                stage.set_initial(stage.t, self.time_grid*stage._T.T_init)
            else:
                stage.set_initial(stage.t, self.time_grid*stage._T)
        if not isinstance(stage._T, FreeTime): self.time_grid*= stage._T
        if not isinstance(stage._t0, FreeTime): self.time_grid+= stage._t0
        self.control_grid = MX(stage.t0 + self.normalized_time_grid*stage.T)

        if not isinstance(stage._T, FreeTime) and not isinstance(stage._t0, FreeTime):
            dT = stage._T/(self.N)
        else:
            dT = self.T/(self.N)
            stage.subject_to(stage.at_t0(self.T)>0)
            stage.set_initial(self.T, stage._T.T_init)

        # f = stage._ode()
        f = SamplingMethod(intg = self.intg, M = self.M).discrete_system(stage).expand()
        self.f = f

        xkp1 = MX.sym('xkp1', stage.nx)
        x_sym = stage.x # MX.sym('x', stage.nx)
        u_sym = stage.u #MX.sym('u', stage.nu)
        global_params_sym = stage.p #MX.sym('p', stage.np)
        stage_params_sym = veccat(*(stage.parameters['control'] + stage.parameters['control+']))
        self.stage_params_sym = stage_params_sym
        global_params_sym = veccat(*stage.parameters[''])
        self.global_params_sym = global_params_sym
        x_next = f(x0=x_sym, u=u_sym, T=dT, p=stage.p)["xf"]
        self.x_next = x_next
        b_x = - xkp1 + x_next

        options = {}
        options["with_header"] = True
        self.codegen = CodeGenerator(f"{self.codegen_name}.c", options)

        BAbt = cs.jacobian(x_next, veccat(u_sym, x_sym)).T
        BAbt = cs.vertcat(BAbt, b_x.T)

        self.artifacts.append(SourceArtifact(f"{self.codegen_name}.c"))
        self.artifacts.append(SourceArtifact(f"{self.codegen_name}.h"))
        self.artifacts.append(LibraryArtifact("fatrop_driver", "lib"))


        # self.artifacts.append(SourceArtifact(f"{self.fatrop_driver}.cpp"))
        # self.wrapper = Wrapper(userparam=self.user_fatrop)

        assert len(stage.variables['control'])==0, "variables defined on control grid not supported. Use controls instead."
        self.v = cs.MX.zeros(0,1)

        self.p_global = cs.vvcat(stage.parameters[''])
        self.p_stage = stage.parameters['control'] + stage.parameters['control+']
        self.X_gist = [cs.vvcat([MX.sym("Xg", s.size1(), s.size2()) for s in stage.states]) for k in range(self.N+1)]
        self.U_gist = [cs.vvcat([MX.sym("Ug", s.size1(), s.size2()) for s in stage.controls]) for k in range(self.N)]
        self.P_stage_gist = [cs.vvcat([MX.sym("stage_param", s.size1(), s.size2()) for s in  self.p_stage]) for k in range(self.N+1)]
        self.P_global_gist = cs.vvcat([MX.sym("Vg", s.size1(), s.size2()) for s in stage.parameters['']])
        self.T_gist = MX.sym("Tg")

        self.x = cs.vvcat(self.X_gist+self.U_gist)


        self.X = self.opti.variable(*stage.x.shape)
        self.U = self.opti.variable(*stage.u.shape)
        self.V = self.opti.variable(*self.v.shape)
        self.P = self.opti.parameter(*stage.p.shape)
        self.t = self.opti.parameter()
        self.T = self.opti.variable()

        self.raw = [stage.x,stage.u,stage.p,stage.t, self.v]
        self.optivar = [self.X, self.U, self.P, self.t, self.V]
        if self.free_time:
            self.raw += [stage.T]
            self.optivar += [self.T]

        

        inits = []
        inits.append((stage.T, stage._T.T_init if isinstance(stage._T, FreeTime) else stage._T))
        inits.append((stage.t0, stage._t0.T_init if isinstance(stage._t0, FreeTime) else stage._t0))

        self.inits = inits

        self.control_grid_init = evalf(substitute([self.control_grid], [a for a,b in inits],[b for a,b in inits])[0])

        #self.control_grid = self.normalized_time_grid

        self.lagrange = MX(0)
        self.mayer = MX(0)
        self.objI = MX(0)
        var_mayer = []
        obj = MX(stage._objective)
        terms = get_terms(obj)
        for term in terms:
            n = [e.name() for e in symvar(term)]
            sumi = np.sum([e=="r_sum_control" for e in n])
            sumip = np.sum([e=="r_sum_control_plus" for e in n])
            summ = np.sum([e=="r_at_tf" for e in n])
            summI = np.sum([e=="r_at_t0" for e in n])
            if sumi + sumip + summ  + summI!= 1:
                raise Exception("Objective cannot be parsed")
            if sumi==1 or sumip ==1:
                n_hits = 0
                for e,parents in visit(term):
                    if e.is_symbolic() and e.name()=="r_sum_control":
                        n_hits+=1
                        for pi in range(len(parents)):
                            p = parents[pi]
                            pchild = parents[pi-1] if pi>0 else e
                            correct = False
                            # Distributive operations
                            if p.op()==cs.OP_MUL:
                                correct = True
                            if p.op()==cs.OP_DIV:
                                # Only allow integral in LHS
                                correct = hash(p.dep(0))==hash(pchild)
                            assert correct, "Objective cannot be parsed: integrals can only be multiplied or divided."
                    if n_hits>1:
                        raise Exception("Objective cannot be parsed")
                if summ!=0:
                    raise Exception("Objective cannot be parsed: operation combining integral and at_tf part not supported")
                self.lagrange += term
                self.objI += term
            if sumip==1 or summ ==1:
                self.mayer += term
            if summI ==1:
                self.objI += term 
        self.P0 = DM.zeros(stage.np)

    def discrete_system(self, stage):
        # Added stage as argument to make it compatible with Stage.discrete_system()
        return self.f

    def set_value(self, stage, master, parameter, value):
        value = evalf(value)
        if parameter in stage.parameters['']:
            p = vec(parameter)
            fill_in(p, self.global_params_sym, value, self.global_params_value,0)
        else:
            p = vec(parameter)
            # for k in list(range(self.N)):
            if p.numel()==value.numel():
                if(value.shape[1] == p.shape[0]):
                    value = value.T
                value = cs.repmat(value,  1, self.N+1)
            if p.numel()*(self.N)==value.numel() or p.numel()*(self.N+1)== value.numel():
                if(value.shape[1] == p.shape[0]):
                    value = value.T
            assert(value.shape[0] == p.shape[0])
            if p.numel()*(self.N)== value.numel():
                value = cs.horzcat(value, value[:,-1])
            fill_in_array(p, self.stage_params_sym, value, self.stage_params_value)


        ## TODO this seems to work but is this the right way to do it?
        # master.stage._param_vals[parameter] = value
    def get_parameters(self, stage):
        self.global_params_value = DM.zeros(self.global_params_sym.shape)
        self.stage_params_value = DM.zeros(self.stage_params_sym.shape[0], self.N+1)
        for i, p in enumerate(stage.parameters['']):
            self.set_value(stage, None, p, stage._param_value(p))
        for i, p in enumerate(stage.parameters['control'] + stage.parameters['control+']):
            self.set_value(stage,None,  p, stage._param_value(p))
        return self.global_params_value, self.stage_params_value
    def set_initial(self, arg1, arg2, initial_dict):
        self.U0, self.X0 = self.get_initial(self.stage, initial_dict)
        if self.mode == "fatropy":
            self.myOCP.set_initial_u(self.U0)
            self.myOCP.set_initial_x(self.X0)
        pass
    def add_sampler(self, name, expr):
        self.samplers[name] = expr

    def _register(self,fun_name,argtypes,restype):
        self.prefix=""
        fun = getattr(self.lib,self.prefix+fun_name)
        setattr(self,"_"+fun_name,fun)
        fun.argtypes = argtypes
        fun.restype = restype

    def get_initial(self, stage, initial_dict):

        U0= DM.zeros(stage.nu, self.N)
        X0 = DM.zeros(stage.nx, self.N+1)
        for var, expr in initial_dict.items():
                # print(var, expr)
                var = vec(var)
                expr = expr
                value = evalf(expr)
                # value = vec(value)
                if depends_on(var,stage.u):
                    assert not depends_on(var, self.v)
                    if var.numel()==value.numel():
                        value = vec(value)
                        if(value.shape[1] == var.shape[0]):
                            value = value.T
                        value = cs.repmat(value,  1, self.N)
                    if var.numel()*(self.N)==value.numel():
                        if(value.shape[1] == var.shape[0]):
                            value = value.T
                    if var.numel()*(self.N+1)==value.numel():
                        if(value.shape[1] == var.shape[0]):
                            value = value.T
                        value = value[:,:-1]
                    assert(value.shape[0] == var.shape[0])
                    fill_in_array(var, stage.u, value, U0)
                if depends_on(var,stage.x):
                    assert not depends_on(vec(var), self.v)
                    if var.numel()==value.numel():
                        value = vec(value)
                        if(value.shape[1] == var.shape[0]):
                            value = value.T
                        value = cs.repmat(value,  1, self.N+1)
                    if var.numel()*(self.N+1)==value.numel():
                        if(value.shape[1] == var.shape[0]):
                            value = value.T
                    assert(value.shape[0] == var.shape[0])
                    fill_in_array(var, stage.x, value, X0)
        return U0, X0

    # def eval_expr(expr, )
    def transcribe_phase2(self, stage, build_dir_abs=None,**kwargs):

        if build_dir_abs is None: build_dir_abs = self.build_dir_abs
        print("build_dir_abs", build_dir_abs, kwargs)
        # Phase 2 adds the constraints
        
        opti_advanced = self.opti.advanced
        placeholders = kwargs["placeholders"]



        # Total Lagrange integrand
        lagrange = placeholders(self.lagrange,preference=['expose'])
        # Total Mayer term
        mayer = placeholders(self.mayer,preference=['expose'])
        objI = placeholders(self.objI,preference=['expose'])
        self.x_next = placeholders(self.x_next, preference='expose')

        x0_eq = []
        x0_b = []

        eq_init = [MX.zeros(0)]
        eq_mid = [MX.zeros(0)] #
        eq_term = [MX.zeros(0)]
        ineq_init = [MX.zeros(0)]
        ineq_mid = [MX.zeros(0)] # <=0
        ineq_term = [MX.zeros(0)]
        ub_init = [MX.zeros(0)]
        lb_init = [MX.zeros(0)]
        ub_mid = [MX.zeros(0)]
        lb_mid = [MX.zeros(0)]
        ub_term = [MX.zeros(0)]
        lb_term = [MX.zeros(0)]  

        # Process initial point constraints
        # Probably should de-duplicate stuff wrt path constraints code
        for c, meta, _ in stage._constraints["point"]:
            # Make sure you resolve u to r_at_t0/r_at_tf

            if not 'r_at_t0' in [a.name() for a in symvar(c)]:
                continue
            c = placeholders(c,max_phase=1)

            cb = c
            c = substitute([placeholders(c,preference='expose')],self.raw,self.optivar)[0]
            mc = opti_advanced.canon_expr(c) # canon_expr should have a static counterpart
            lb,canon,ub = substitute([mc.lb,mc.canon,mc.ub],self.optivar,self.raw)
            check = not depends_on(canon, veccat(self.v))
            assert check, 'v variables are not supported yet'
            if mc.type == casadi.OPTI_EQUALITY:
                eq_init.append(canon-ub)
            else:
                assert mc.type in [casadi.OPTI_INEQUALITY, casadi.OPTI_GENERIC_INEQUALITY, casadi.OPTI_DOUBLE_INEQUALITY]
                ineq_init.append(canon)
                ub_init.append(ub)
                lb_init.append(lb)
        
        # Process terminal point constraints
        # Probably should de-duplicate stuff wrt path constraints code
        for c, meta, _ in stage._constraints["point"]:
            # Make sure you resolve u to r_at_t0/r_at_tf

            if not 'r_at_tf' in [a.name() for a in symvar(c)]:
                continue
            c = substitute([placeholders(c,preference='expose')],self.raw,self.optivar)[0]
            mc = opti_advanced.canon_expr(c) # canon_expr should have a static counterpart
            lb,canon,ub = substitute([mc.lb,mc.canon,mc.ub],self.optivar,self.raw)

            check = not depends_on(canon, veccat(stage.u, self.v))
            assert check, "at t=tF, only constraints on x are allowed. Got '%s'" % str(c)

            if mc.type == casadi.OPTI_EQUALITY or mc.type == casadi.OPTI_GENERIC_EQUALITY:
                eq_term.append(canon-ub)
            else:
                assert mc.type in [casadi.OPTI_INEQUALITY, casadi.OPTI_GENERIC_INEQUALITY, casadi.OPTI_DOUBLE_INEQUALITY]
                ineq_term.append(canon)
                ub_term.append(ub)
                lb_term.append(lb)
        

        # Process path constraints
        # TODO check for include first and include last
        for c, meta, args in stage._constraints["control"]+stage._constraints["integrator"]:
            c = substitute([placeholders(c,preference=['expose'])],self.raw,self.optivar)[0]
            mc = opti_advanced.canon_expr(c) # canon_expr should have a static counterpart
            lb,canon,ub = substitute([mc.lb,mc.canon,mc.ub],self.optivar,self.raw)
            check = not depends_on(canon, veccat(self.v))
            assert check, 'v variables'
            # lb <= canon <= ub
            # Check for infinities
            try:
                lb_inf = np.all(np.array(evalf(lb)==-inf))
            except:
                lb_inf = False
            try:
                ub_inf = np.all(np.array(evalf(ub)==inf))
            except:
                ub_inf = False

            if mc.type == casadi.OPTI_EQUALITY:
                eq_mid.append(canon-ub)
                if args['include_first']:
                    eq_init.append(canon-ub)
                if args['include_last']:
                    check = not depends_on(canon, veccat(stage.u, self.v))
                    if not check and not depends_on(canon, veccat(stage.x)): continue
                    assert check, "at t=tF, only constraints on x are allowed. Got '%s'" % str(c)
                    eq_term.append(canon-ub)
            else:
                assert mc.type in [casadi.OPTI_INEQUALITY, casadi.OPTI_GENERIC_INEQUALITY, casadi.OPTI_DOUBLE_INEQUALITY]

                ineq_mid.append(canon)
                ub_mid.append(ub)
                lb_mid.append(lb)
                if args['include_first']:
                    ineq_init.append(canon)
                    ub_init.append(ub)
                    lb_init.append(lb)
                if args['include_last']:
                    check = not depends_on(canon, veccat(stage.u, self.v))
                    if not check and not depends_on(canon, veccat(stage.x)): continue
                    assert check, "at t=tF, only constraints on x are allowed. Got '%s'" % str(c)
                    ineq_term.append(canon)
                    ub_term.append(ub)
                    lb_term.append(lb)
        


        #### parameters
        global_params_value, stage_params_value =self.global_params_value, self.stage_params_value = self.get_parameters(stage)

        #### initializiaton
        U0, X0 = self.U0, self.X0 = self.get_initial(stage, stage._initial)


        eqI = vvcat(eq_init)
        eq = vvcat(eq_mid)
        eqF = vvcat(eq_term)
        ineqI = vvcat(ineq_init)
        ineq = vvcat(ineq_mid)
        ineqF = vvcat(ineq_term)
        lb_init = vvcat(lb_init)
        ub_init = vvcat(ub_init)
        lb_mid = vvcat(lb_mid)
        ub_mid = vvcat(ub_mid)
        ub_term = vvcat(ub_term)
        lb_term = vvcat(lb_term)

        ngI = eqI.shape[0]
        ng = eq.shape[0]
        ngF = eqF.shape[0]
        ngIneqI = ineqI.shape[0]
        ngIneq = ineq.shape[0]
        ngIneqF = ineqF.shape[0]
        dual_dyn = MX.sym('d_dyn', stage.nx)
        dual_eqI = MX.sym('d_eqI', ngI)
        dual_eq = MX.sym('d_eq', ng)
        dual_eqF = MX.sym('d_eqF', ngF)
        dualIneqI = MX.sym('d_IneqI', ngIneqI)
        dualIneq = MX.sym('d_Ineq', ngIneq)
        dualIneqF = MX.sym('d_IneqF', ngIneqF)
        nx = stage.nx
        nu = stage.nu
        def expand_help(func):
            if self.expand:
                return func.expand()
            else:
                return func


        #TODO: Stagewise equality constraints seem to be missing in FATROP?

        obj_scale = MX.sym('obj_scale')
                # BAbt
        stateskp1 = MX.sym("states_kp1", nx)
        BAbt = MX.zeros(nu+nx+1, nx)
        BAbt[:nu+nx,
             :] = jacobian(self.x_next, veccat(stage.u, stage.x)).T
        b = (-stateskp1 + self.x_next)[:]
        BAbt[nu+nx, :] = b
        self.codegen.add(
            expand_help(Function("BAbt", [stateskp1, stage.u, stage.x, self.stage_params_sym, self.global_params_sym], [densify(BAbt)])))
        # b
        self.codegen.add(expand_help(Function("bk", [stateskp1, stage.u,
                              stage.x, self.stage_params_sym, self.global_params_sym], [densify(b)])))
        # RSQrqtI
        RSQrqtI = MX.zeros(nu+nx+1, nu + nx)
        [RSQI, rqI] = cs.hessian(objI, veccat(stage.u, stage.x))
        RSQIGN = RSQI
        rqlagI = rqI
        if ngI > 0:
            [H, h]= cs.hessian(dual_eqI.T@eqI, veccat(stage.u, stage.x))
            RSQI += H
            rqlagI += h
        [H,h] = cs.hessian(dual_dyn.T@self.x_next,
                        veccat(stage.u, stage.x))
        RSQI += H
        rqlagI += h
        
        if ngIneqI > 0:
            [H,h] = cs.hessian(dualIneqI.T@ineqI,
                            veccat(stage.u, stage.x))
            RSQI += H
            rqlagI += h
        RSQrqtI[:nu+nx, :] = RSQI
        RSQrqtI[nu+nx, :] = rqlagI[:]
        self.codegen.add(expand_help(Function("RSQrqtI", [obj_scale, stage.u,
              stage.x, dual_dyn, dual_eqI, dualIneqI, self.stage_params_sym, self.global_params_sym], [densify(RSQrqtI)])))
        RSQrqtI[:nu+nx, :] = RSQIGN
        RSQrqtI[nu+nx, :] = rqlagI[:]
        self.codegen.add(expand_help(Function("RSQrqtIGN", [obj_scale, stage.u,
              stage.x, dual_dyn, dual_eqI, dualIneqI, self.stage_params_sym, self.global_params_sym], [densify(RSQrqtI)])))
        # rqI
        self.codegen.add(expand_help(Function("rqI", [obj_scale,
              stage.u, stage.x, self.stage_params_sym, self.global_params_sym], [densify(rqI)])))
        # RSQrqt
        RSQrqt = MX.zeros(nu+nx+1, nu + nx)
        [RSQ, rq] = cs.hessian(lagrange, veccat(stage.u, stage.x))
        RSQGN = RSQ
        rqlag = rq
        if ng > 0:
            [H, h]= cs.hessian(dual_eq.T@eq, veccat(stage.u, stage.x))
            RSQ += H
            rqlag += h
        [H,h]= cs.hessian(dual_dyn.T@self.x_next,
                       veccat(stage.u, stage.x))
        RSQ += H
        rqlag +=h

        if ngIneq > 0:
            [H,h] = cs.hessian(dualIneq.T@ineq,
                           veccat(stage.u, stage.x))
            RSQ += H
            rqlag +=h
        RSQrqt[:nu+nx, :] = RSQ
        RSQrqt[nu+nx, :] = rqlag[:]
        self.codegen.add(expand_help(Function("RSQrqt", [obj_scale, stage.u, stage.x,
              dual_dyn, dual_eq, dualIneq, self.stage_params_sym, self.global_params_sym], [densify(RSQrqt)])))
        RSQrqt[:nu+nx, :] = RSQGN
        RSQrqt[nu+nx, :] = rqlag[:]
        self.codegen.add(expand_help(Function("RSQrqtGN", [obj_scale, stage.u, stage.x,
              dual_dyn, dual_eq, dualIneq, self.stage_params_sym, self.global_params_sym], [densify(RSQrqt)])))
        # rqF
        self.codegen.add(expand_help(Function("rqk", [obj_scale,
              stage.u, stage.x, self.stage_params_sym, self.global_params_sym], [densify(rq)])))
        # Lk
        self.codegen.add(expand_help(Function("LI", [obj_scale, stage.u,
              stage.x, self.stage_params_sym, self.global_params_sym], [densify(objI)])))
        # Lk
        self.codegen.add(expand_help(Function("Lk", [obj_scale, stage.u,
              stage.x, self.stage_params_sym, self.global_params_sym], [densify(lagrange)])))
        # RSQrqtF
        RSQrqtF = MX.zeros(nx+1, nx)
        [RSQF, rqF] = cs.hessian(mayer, veccat(stage.x))
        RSQFGN = RSQF
        rqlagF = rqF
        if ngF > 0:
            [H, h]= cs.hessian(dual_eqF.T@eqF,
                            veccat(stage.x))
            RSQF += H
            rqlagF += h
        if ngIneqF > 0:
            [H,h] = cs.hessian(dualIneqF.T@ineqF,
                           veccat(stage.x))
            RSQF += H
            rqlagF += h
        # if ngIneq>-1:
        #     RSQF += cs.hessian(dualIneq.T@ineq, vertcat(stage.u, stage.x))[-1]
        RSQrqtF[:nx, :] = RSQF
        RSQrqtF[nx, :] = rqlagF[:]
        self.codegen.add(expand_help(Function("RSQrqtF", [obj_scale, stage.u, stage.x,
              dual_dyn, dual_eqF, dualIneqF, self.stage_params_sym, self.global_params_sym], [densify(RSQrqtF)])))
        RSQrqtF[:nx, :] = RSQFGN
        RSQrqtF[nx, :] = rqlagF[:]
        self.codegen.add(expand_help(Function("RSQrqtFGN", [obj_scale, stage.u, stage.x,
              dual_dyn, dual_eqF, dualIneqF, self.stage_params_sym, self.global_params_sym], [densify(RSQrqtF)])))
        # rqF
        self.codegen.add(expand_help(Function("rqF", [obj_scale,
              stage.u, stage.x, self.stage_params_sym, self.global_params_sym], [densify(rqF)])))
        # LF
        self.codegen.add(expand_help(Function("LF", [obj_scale, stage.u,
              stage.x, self.stage_params_sym, self.global_params_sym], [densify(mayer)])))
        # GgtI
        GgtI = MX.zeros(nu+nx+1, ngI)
        GgtI[:nu+nx,
             :] = jacobian(eqI, veccat(stage.u, stage.x)).T
        GgtI[nu+nx, :] = eqI[:].T
        self.codegen.add(expand_help(Function(
            "GgtI", [stage.u, stage.x, self.stage_params_sym, self.global_params_sym], [densify(GgtI)])))
        # g_I
        self.codegen.add(expand_help(Function("gI", [stage.u, stage.x, self.stage_params_sym,
              self.global_params_sym], [densify(eqI[:])])))
        # Ggt
        Ggt = MX.zeros(nu+nx+1, ng)
        Ggt[:nu+nx,
             :] = jacobian(eq, veccat(stage.u, stage.x)).T
        Ggt[nu+nx, :] = eq[:].T
        self.codegen.add(expand_help(Function(
            "Ggt", [stage.u, stage.x, self.stage_params_sym, self.global_params_sym], [densify(Ggt)])))
        # g
        self.codegen.add(expand_help(Function("g", [stage.u, stage.x, self.stage_params_sym,
              self.global_params_sym], [densify(eq[:])])))
        # GgtF
        GgtF = MX.zeros(nx+1, ngF)
        GgtF[:nx, :] = jacobian(eqF, stage.x).T
        GgtF[nx, :] = eqF[:].T
        self.codegen.add(expand_help(Function(
            "GgtF", [stage.u, stage.x, self.stage_params_sym, self.global_params_sym], [densify(GgtF)])))
        # g_F
        self.codegen.add(expand_help(Function("gF", [stage.u, stage.x, self.stage_params_sym,
              self.global_params_sym], [densify(eqF[:])])))
        # GgineqIt
        GgineqIt = MX.zeros(nu+nx+1, ngIneqI)
        GgineqIt[:nu+nx,
                :] = jacobian(ineqI, veccat(stage.u, stage.x)).T
        GgineqIt[nu+nx, :] = ineqI[:].T
        self.codegen.add(expand_help(Function("GgineqIt", [stage.u,
              stage.x, self.stage_params_sym, self.global_params_sym], [densify(GgineqIt)])))
        self.codegen.add(expand_help(Function("gineqI", [stage.u, stage.x, self.stage_params_sym, self.global_params_sym], [
              densify(ineqI[:])])))
        # Ggineqt
        Ggineqt = MX.zeros(nu+nx+1, ngIneq)
        Ggineqt[:nu+nx,
                :] = jacobian(ineq, veccat(stage.u, stage.x)).T
        Ggineqt[nu+nx, :] = ineq[:].T
        self.codegen.add(expand_help(Function("Ggineqt", [stage.u,
              stage.x, self.stage_params_sym, self.global_params_sym], [densify(Ggineqt)])))
        self.codegen.add(expand_help(Function("gineq", [stage.u, stage.x, self.stage_params_sym, self.global_params_sym], [
              densify(ineq[:])])))
        # GgineqFt
        GgineqFt = MX.zeros(nx+1, ngIneqF)
        GgineqFt[:nx,
                :] = jacobian(ineqF, veccat(stage.x)).T
        GgineqFt[nx, :] = ineqF[:].T
        self.codegen.add(expand_help(Function("GgineqFt", [
              stage.x, self.stage_params_sym, self.global_params_sym], [densify(GgineqFt)])))
        self.codegen.add(expand_help(Function("gineqF", [stage.x, self.stage_params_sym, self.global_params_sym], [
              densify(ineqF[:])])))
        
        # codegenerate samplers
        for sampl in self.samplers.keys():
            self.codegen.add(expand_help(Function("sampler_"+ str(sampl), [stage.u, stage.x, self.stage_params_sym, self.global_params_sym], [densify(placeholders(self.samplers[sampl], preference =['expose']))])))


        self.build_dir_abs = build_dir_abs
        # prepare_build_dir(self.build_dir_abs)
        # check if the build directory exists otherwise create it
        if not os.path.exists(self.build_dir_abs):
            os.makedirs(self.build_dir_abs)

        self.codegen.generate(self.build_dir_abs+os.sep)
        def get_sym_name(symbol):
            # return symbol.name() if bool(symbol not in self.variable_names) else self.variable_names[symbol]
            return symbol.name()

        # creating the json file for fatrop
        control_params_sym = stage.parameters['control'] + stage.parameters['control+']
        global_params_sym = stage.parameters['']
        control_params_offset = {get_sym_name(sym):get_offsets(vec(sym), veccat(*control_params_sym)) for sym in control_params_sym}
        global_params_offset = {get_sym_name(sym):get_offsets(vec(sym), veccat(*global_params_sym)) for sym in global_params_sym}
        states_sym = [vec(state) for state in  stage.states]
        states_offset = {get_sym_name(sym):get_offsets(vec(sym), veccat(*states_sym)) for sym in stage.states}
        controls_sym = [vec(control) for control in  stage.controls]
        controls_offset = {get_sym_name(sym):get_offsets(vec(sym), veccat(*controls_sym)) for sym in stage.controls}



        json_dict = {
                        'control_params_offset': control_params_offset,
                        'global_params_offset': global_params_offset,
                        'states_offset': states_offset,
                        'controls_offset': controls_offset,
                        'nx': stage.nx,
                        'nu': stage.nu,
                        'ngI': eqI.shape[0],
                        'ng': ng,
                        'ngF': eqF.shape[0],
                        'ng_ineqI': ngIneqI,
                        'ng_ineq': ngIneq,
                        'ng_ineqF': ineqF.shape[0],
                        'n_stage_params': self.stage_params_sym.shape[0],
                        'n_global_params': self.global_params_sym.shape[0],
                        'global_params': global_params_value.T.full().flatten().tolist(), #TODO
                        'stage_params' : stage_params_value.T.full().flatten().tolist(), #TODO
                        'params': [], #TODO
                        'K': self.N+1,
                        'initial_x': X0.T.full().flatten().tolist(),
                        'initial_u': U0.T.full().flatten().tolist(),
                        'lower': cs.repmat(evalf(lb_mid), (self.N-1, 1)).full().flatten().tolist(),
                        'upper': cs.repmat(evalf(ub_mid), (self.N-1, 1)).full().flatten().tolist(),
                        'lowerI': evalf(lb_init).full().flatten().tolist(),
                        'upperI': evalf(ub_init).full().flatten().tolist(),                                               
                        'lowerF': evalf(lb_term).full().flatten().tolist(),
                        'upperF': evalf(ub_term).full().flatten().tolist(),
                        'samplers': list(self.samplers.keys())
                    }
        

        import json
        with open(self.build_dir_abs + os.sep + 'casadi_codegen.json', 'w') as fp:
            json.dump(json_dict, fp, indent = 4)


        class MacroGen:
            def __init__(self):
                self.str = ""
            def process_macro_var(self, macro_name, value):
                self.str += "#define MACRO_" + macro_name + " " + str(value) + "\n"
            def process_macro_array(self, macro_name, value_array):
                self.str += "#define MACRO_" + macro_name + " {" + ", ".join([str(value) if value!=inf else "std::numeric_limits<double>::infinity()" for value in value_array]) + "}\n"
            def __str__(self):
                return self.str
            
        macro_gen = MacroGen()
        macro_gen.process_macro_var('nx', stage.nx)
        macro_gen.process_macro_var('nu', stage.nu)
        macro_gen.process_macro_var('ngI', eqI.shape[0])
        macro_gen.process_macro_var('ng', ng)
        macro_gen.process_macro_var('ngF', eqF.shape[0])
        macro_gen.process_macro_var('ng_ineqI', ngIneqI)
        macro_gen.process_macro_var('ng_ineq', ngIneq)
        macro_gen.process_macro_var('ng_ineqF', ineqF.shape[0])
        macro_gen.process_macro_var('n_stage_params', self.stage_params_sym.shape[0])
        macro_gen.process_macro_var('n_global_params', self.global_params_sym.shape[0])
        macro_gen.process_macro_array('global_params', global_params_value.T.full().flatten().tolist())
        macro_gen.process_macro_array('stage_params' , stage_params_value.T.full().flatten().tolist())
        macro_gen.process_macro_var('K', self.N+1)
        macro_gen.process_macro_array('initial_x', X0.T.full().flatten().tolist())
        macro_gen.process_macro_array('initial_u', U0.T.full().flatten().tolist())
        macro_gen.process_macro_array('bounds_L', evalf(lb_init).full().flatten().tolist() + cs.repmat(evalf(lb_mid), (self.N-1, 1)).full().flatten().tolist() +  evalf(lb_term).full().flatten().tolist())
        macro_gen.process_macro_array('bounds_U',  evalf(ub_init).full().flatten().tolist() + cs.repmat(evalf(ub_mid), (self.N-1, 1)).full().flatten().tolist() +  evalf(ub_term).full().flatten().tolist())
        macro_gen.process_macro_array('samplers', list(self.samplers.keys()))

        # write out the macro_gen to a file
        with open(self.build_dir_abs + '/problem_information.h', 'w') as fp:
            fp.write(str(macro_gen))

        functions_c = str(self.build_dir_abs + "/casadi_codegen.c")
        json_spec = str(self.build_dir_abs + "/casadi_codegen.json")

        if self.mode == "fatropy":
            import fatropy
            if platform.system()=="Windows":
                functions = str(self.build_dir_abs + "/casadi_codegen.dll")
                cmd = f'"{vcvars}" && cl /LD /O2 /Fe: ' + functions + ' ' +  functions_c
                print(cmd)
                subprocess.run(cmd,shell=True)
            else:
                functions = str(self.build_dir_abs + "/casadi_codegen.so")
                subprocess.run("gcc -fPIC -march=native -shared -Ofast " + functions_c + " -o " + functions, shell = True)
            # self.myOCP = fatropy.OCP(functions,json_spec)
            self.myOCP = fatropy.StageOCPApplicationFactory.from_rockit_interface(functions, json_spec)

        elif self.mode == "interface":
            # check if interface_generation directory exists
            if not os.path.exists(self.build_dir_abs + "/interface/build"):
                # copy the interface_generation directory of this files path to the current directory
                shutil.copytree(os.path.dirname(os.path.realpath(__file__)) + os.sep + "interface_generation",self.build_dir_abs + "/interface/")
            
            with open(self.build_dir_abs + "/interface/after_init.c.in", "w") as after_init:
                for name, value in self.fatrop_options.items():
                    after_init.write(f"""set_option("{name}",{value});\n""")
            # run cmake to build the interface
            subprocess.run(["cmake","-DCMAKE_BUILD_TYPE=Release","-DWITH_PYTHON=OFF","-S", "interface","-B", os.path.join("interface","build")], cwd=self.build_dir_abs)
            # run make to build the interface
            subprocess.run(["cmake","--build","interface/build","--config","Release"], cwd=self.build_dir_abs)
            subprocess.run(["cmake","--install","interface/build","--prefix","."], cwd=self.build_dir_abs)
            self.fatrop_solver = cs.external("fatrop_driver", self.build_dir_abs + "/lib/libfatrop_driver.so")
            # self.solver.checkout()
            # use ctypes to call
            # void set_option(const std::string& option, const double& value)
            # from libfatrop_driver.so
            import ctypes
            # Declare the function prototype
            self.lib = ctypes.PyDLL(self.build_dir_abs + "/lib/libfatrop_driver.so")
            set_option_argtypes = [ctypes.c_char_p, ctypes.c_double]
            self._register("set_option", set_option_argtypes, None)
            pass 

        return
    
    def set_option(self, option, value):
        if self.mode == "fatropy":
            self.myOCP.set_option(option, value)
        elif self.mode == "interface":
            self._set_option(option.encode('utf-8'), value)
        return
    
    def solver(self, name , opts: dict):
        for key, value in opts.items():
            self.set_option(key, value)
        return




    def to_function(self, stage, name, args, results, *margs):
        args = [stage.value(a) for a in args]
        class FatropCallBack(Callback):
            def __init__(self,solver, n_stage_params, n_global_params,nu, nx, K):
                Callback.__init__(self)
                self.fatrop_solver = solver
                self.nx = nx 
                self.nu = nu 
                self.K = K 
                self.n_stage_params = n_stage_params
                self.n_global_params = n_global_params
                self.construct("fatrop_callback", {})
            def get_n_in(self):
                return 4 # stage_params, global_params, u, x
            def get_n_out(self):
                return 2 # u_opt, x_opt
            def get_sparsity_in(self, input):
                if(input ==0):
                    return cs.Sparsity.dense(self.n_stage_params, self.K) 
                if(input ==1):
                    return cs.Sparsity.dense(self.n_global_params, 1) 
                if(input ==2):
                    return cs.Sparsity.dense(self.nu, self.K-1) 
                if(input ==3):
                    return cs.Sparsity.dense(self.nx, self.K) 
                return
                # return super().get_sparsity_in(*args)
            def get_sparsity_out(self, output):
                if(output ==0):
                    return cs.Sparsity.dense(self.nu, self.K-1) 
                if(output ==1):
                    return cs.Sparsity.dense(self.nx, self.K) 
                return
            def init(self):
                return
            
            def eval(self, arg):
                self.fatrop_solver.set_params(arg[0], arg[1])
                self.fatrop_solver.set_initial_u(cs.DM(arg[2]))
                self.fatrop_solver.set_initial_x(cs.DM(arg[3]))
                self.fatrop_solver.optimize()
                sol = self.fatrop_solver.last_solution()
                return [cs.DM(sol.u), cs.DM(sol.x)] 
            
            def has_codegen(sefl):
                return True
            
        controls = cs.hcat(self.U_gist)
        states = cs.hcat(self.X_gist)
        p_stage = cs.hcat(self.P_stage_gist)
        p_global = self.P_global_gist

        gist_list = self.U_gist+self.X_gist+self.P_stage_gist+[self.P_global_gist,self.T_gist]

        try:
            helper0 = Function("helper0", args, gist_list, {"allow_free":True})
        except:
            helper0 = Function("helper0", args, gist_list)

        if helper0.has_free():
            helper0_free = helper0.free_mx()
            print("test",helper0_free, self.initial_value(stage, helper0_free))
            [controls, states, p_stage, p_global, T_gist] = cs.substitute([controls, states, p_stage, p_global, self.T_gist], helper0_free, self.initial_value(stage, helper0_free))
        if self.mode == "fatropy":
            self.fatrop_solver = FatropCallBack(self.myOCP, self.P_stage_gist[0].size1(), self.P_global_gist.size1(), stage.nu, stage.nx, self.N + 1)
            res_u, res_x = self.fatrop_solver(p_stage, p_global, controls, states)
        elif self.mode == "interface":
            res_x, res_u, _, _ = self.fatrop_solver(densify(p_global.monitor("p_global")), densify(p_stage.monitor("p_stage")), densify(states), densify(controls))

        ret = Function(name, args, cs.substitute(results, gist_list, cs.horzsplit(res_u)+cs.horzsplit(res_x)+cs.horzsplit(p_stage)+[p_global,T_gist]) , *margs)
        assert not ret.has_free()
        return ret

    def initial_value(self, stage, expr):
        # check if expr is a list
        lst = isinstance(expr, list) 
        if not lst:
            expr = [expr]
        # expr = stage.value(expr)
        [_,states] = stage.sample(stage.x,grid='control')
        [_,controls] = stage.sample(stage.u,grid='control-')
        variables = stage.value(veccat(stage.variables['']))

        helper_in = [self.P_global_gist,cs.horzcat(*self.P_stage_gist),states,controls,variables,self.T_gist]
        helper = Function("helper", helper_in, expr)
        ret = [s.toarray(simplify=True) for s in helper.call([self.global_params_value, self.stage_params_value, self.X0, self.U0, 0, self.inits[0][1]])]
        if not lst :
            return ret[0]
        return ret

    def solve(self, stage,limited=False):
        # self.global_params_value, self.stage_params_value = self.get_parameters(stage)
        if self.mode == "fatropy":
            self.myOCP.set_params(self.stage_params_value, self.global_params_value)
            retval = self.myOCP.optimize()
            sol = self.myOCP.last_solution()
            u_sol_fatrop = sol.u 
            x_sol_fatrop = sol.x
        elif self.mode == "interface":
            x_sol_fatrop, u_sol_fatrop, _, _= self.fatrop_solver(self.global_params_value, self.stage_params_value, self.X0, self.U0)



        return OcpSolution(SolWrapper(self, vec(x_sol_fatrop), veccat(vec(u_sol_fatrop), DM.zeros(1)), DM.zeros(0), DM.zeros(0),cs.vertcat(self.stage_params_value), cs.vertcat(self.global_params_value), rT=stage.T), stage)

    def get_stats(self):
        stats = self.stats
        return dict((k,getattr(stats,k)) for k,_ in stats_fields)

    def non_converged_solution(self, stage):
        return self.last_solution

    def solve_limited(self, stage):
        return self.solve(stage,limited=True)

    def eval(self, stage, expr):
        placeholders = stage.placeholders_transcribed
        expr = placeholders(expr,max_phase=1)
        ks = [cs.vvcat(stage.parameters['']), stage.T]
        vs = [self.P_global_gist, self.T_gist]
        ret = substitute([expr],ks,vs)[0]
        return ret
        
    @property
    def gist(self):
        return vertcat(ExternalMethod.gist.fget(self), self.P_global_gist, *self.P_stage_gist, self.T_gist)

    def eval_at_control(self, stage, expr, k):
        placeholders = stage.placeholders_transcribed
        expr = placeholders(expr,max_phase=1)
        ks = [stage.x,stage.u,self.p_global,stage.T, cs.vvcat(stage.parameters['control'] + stage.parameters['control+'])]
        vs = [self.X_gist[k], self.U_gist[min(k, self.N-1)], self.P_global_gist, self.T_gist, self.P_stage_gist[k]]
        if not self.t_state:
            ks += [stage.t]
            vs += [self.control_grid[k]]
        ret = substitute([expr],ks,vs)[0]
        return ret

class SolWrapper:
    def __init__(self, method, x, u, v, T,P_stage, P_global, rT=None):
        self.method = method
        self.x = x
        self.u = u
        self.T = T
        self.v = v
        self.rT = rT
        self.P_stage = P_stage
        self.P_global = P_global

    def value(self, expr, *args,**kwargs):
        placeholders = self.method.stage.placeholders_transcribed
        expr = substitute(expr,self.rT, self.T)
        ret = evalf(substitute([placeholders(expr)],[self.method.gist],[veccat(self.x, self.u, self.v,self.P_stage, self.P_global ,self.T)])[0])
        return ret.toarray(simplify=True)

    def stats(self):
        return self.method.get_stats()
