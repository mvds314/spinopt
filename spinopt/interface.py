# -*- coding: utf-8 -*-

import nlopt
import numpy as np


class NLOptimizer:
    """
    A wrapper around the NLOPT optimizer, that takes only Scipy-style constraints and objective functions as input.

    Minimum input should be:
    - objective function
    - some value that can be fed to the objective function, and constraints. This value is used to determine the dimension.
    """

    def __init__(self, objective_func, x0, **kwargs):
        kwargs.setdefault("self.backend", "slsqp")
        kwargs.setdefault("xtol_rel", 1e-8)
        kwargs.setdefault("xtol_abs", 1e-8)
        kwargs.setdefault("ftol_rel", 1e-8)
        kwargs.setdefault("ftol_abs", 1e-8)
        kwargs.setdefault("maxeval", 10000)
        kwargs.setdefault("maxtime", 300)
        kwargs.setdefault("global_lb", None)
        kwargs.setdefault("global_ub", None)
        kwargs.setdefault("constraints", None)
        self._kwargs = kwargs
        assert isinstance(objective_func, callable)
        self.objective_func = objective_func
        self.x0 = np.asanyarray(x0)

    @property
    def N(self):
        if not hasattr(self, "_N"):
            self._N = len(self.x0)
        return self._N

    @property
    def backend(self):
        return self._kwargs["self.backend"]

    @property
    def xtol_rel(self):
        return self._kwargs["xtol_rel"]

    @property
    def xtol_abs(self):
        return self._kwargs["xtol_abs"]

    @property
    def ftol_rel(self):
        return self._kwargs["ftol_rel"]

    @property
    def ftol_abs(self):
        return self._kwargs["ftol_abs"]

    @property
    def maxeval(self):
        return self._kwargs["maxeval"]

    @property
    def maxtime(self):
        return self._kwargs["maxtime"]

    @property
    def global_lb(self):
        return self._kwargs["global_lb"]

    @property
    def global_ub(self):
        return self._kwargs["global_ub"]

    @property
    def constraints(self):
        return self._kwargs["constraints"]

    @constraints.setter
    def constraints(self, constraints):
        self._kwargs["constraints"] = constraints
        if hasattr(self, "_opt"):
            del self._opt

    @property
    def constraint_tol(self):
        return self._kwargs.get("constraint_tol", self.xtol_abs / 10)

    @staticmethod
    def convert_1d_constraint(con):
        """
        Convert a one-dimensional Scipy constraint, i.e., a constraints that maps to a scalar, to NLOPT format
        """

        def f(x, grad=None):
            args = con["args"] if "args" in con.keys() else []
            if grad is not None and grad.size > 0:
                if "jac" in con.keys():
                    grad[:] = -con["jac"](x, *args)
                else:
                    raise NotImplementedError("Gradient should be specified ")
            return -con["fun"](
                x, *args
            )  # sign accounts for difference between geq 0 and leq 0 constraints

    @staticmethod
    def convert_nd_constraint(con):
        """
        Convert a multi-dimensional Scipy constraint, i.e., a constraints that maps to a Rn, to NLOPT format
        """

        def f(result, x, grad=None):
            args = con["args"] if "args" in con.keys() else []
            if grad is not None and grad.size > 0:
                if "jac" in con.keys():
                    grad[:] = -con["jac"](x, *args)
                else:
                    raise NotImplementedError("Gradient should be specified")
            result[:] = -con["fun"](x, *args)

    def _add_constraints(self):
        """
        Adds constraints in scipy format, i.e., as a list of dictionaries with fields:

        type: str
            Constraint type: ‘eq’ for equality, ‘ineq’ for inequality.
        fun: callable
            The function defining the constraint.
        jac: callable, optional
            The Jacobian of fun.
        args: sequence, optional
            Extra arguments to be passed to the function and Jacobian.
        See also https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize.

        Parameters
        ----------
        opt : NLOPT optimization problem
            object returned by nlopt.opt(...)
        x : numpy array (defautls to x0)
            can be used to evaluate constraints to distinguish between univariate
            and multivariate constraints
        constraints : dict or list or tuple with dictionaries
            containing at least the keys
        constraint_tol : float defauts to self.constraint_tol
            tolerance with which NLOPT requires a constraint to be satisfied

        Returns
        -------
        opt
        """
        for con in self.constraints:
            assert isinstance(con, dict), "Constraints should be dictionaries"
            assert {"type", "fun"}.issubset(con.keys())
            args = con.get("args", [])
            y = con["fun"](self.x0, *args)
            if isinstance(y, float):
                f = self.convert_1d_constraint(con)
                if con["type"] == "ineq":
                    self.opt.add_inequality_constraint(f, self.constraint_tol)
                elif con["type"] == "eq":
                    self.opt.add_equality_constraint(f, self.constraint_tol)
            else:
                y = np.array(y)
                f = self.convert_nd_constraint(con)
                if con["type"] == "ineq":
                    self.opt.add_inequality_mconstraint(
                        f, np.full(y.shape[-1], self.constraint_tol)
                    )
                elif con["type"] == "eq":
                    self.opt.add_equality_mconstraint(f, np.full(y.shape[-1], self.constraint_tol))

    @property
    def get_nlopt_optimizer(self):
        if not hasattr(self, "_opt"):
            local_opt = None
            if self.backend.lower() == "cobyla":
                opt = nlopt.opt(nlopt.LN_COBYLA, self.N)
            elif self.backend.lower() == "slsqp":
                opt = nlopt.opt(nlopt.LD_SLSQP, self.N)
            elif self.backend.lower() == "isres":
                opt = nlopt.opt(nlopt.GN_ISRES, self.N)
            elif self.backend.lower() == "neldermead":
                local_opt = nlopt.opt(nlopt.LN_NELDERMEAD, self.N)
                opt = nlopt.opt(nlopt.AUGLAG, self.N)
            elif self.backend.lower() == "sbplx":
                local_opt = nlopt.opt(nlopt.LN_SBPLX, self.N)
                opt = nlopt.opt(nlopt.AUGLAG, self.N)
            elif self.backend.lower() == "var2":
                local_opt = nlopt.opt(nlopt.LD_VAR2, self.N)
                opt = nlopt.opt(nlopt.AUGLAG, self.N)
            elif self.backend.lower() == "var1":
                local_opt = nlopt.opt(nlopt.LD_VAR1, self.N)
                opt = nlopt.opt(nlopt.AUGLAG, self.N)
            elif self.backend.lower() == "crs":
                local_opt = nlopt.opt(nlopt.GN_CRS2_LM, self.N)
                opt = nlopt.opt(nlopt.AUGLAG, self.N)
            elif self.backend.lower() == "direct_l":
                local_opt = nlopt.opt(nlopt.GN_DIRECT_L, self.N)
                opt = nlopt.opt(nlopt.AUGLAG, self.N)
            elif self.backend.lower() == "stogo":
                local_opt = nlopt.opt(nlopt.GD_STOGO, self.N)
                opt = nlopt.opt(nlopt.AUGLAG, self.N)
            elif self.backend.lower() == "stogo_rand":
                local_opt = nlopt.opt(nlopt.GD_STOGO_RAND, self.N)
                opt = nlopt.opt(nlopt.AUGLAG, self.N)
            elif self.backend.lower() == "ags":
                local_opt = nlopt.opt(nlopt.GN_AGS, self.N)
                opt = nlopt.opt(nlopt.AUGLAG, self.N)
            elif self.backend.lower() == "praxis":
                local_opt = nlopt.opt(nlopt.LN_PRAXIS, self.N)
                opt = nlopt.opt(nlopt.AUGLAG, self.N)
            elif self.backend.lower() == "mma":
                local_opt = nlopt.opt(nlopt.LD_MMA, self.N)
                opt = nlopt.opt(nlopt.AUGLAG_EQ, self.N)
            elif self.backend.lower() == "lbfgs":
                local_opt = nlopt.opt(nlopt.LD_LBFGS, self.N)
                opt = nlopt.opt(nlopt.AUGLAG, self.N)
            else:
                raise Exception("Unsupported NLOPT backend")
            if self.global_lb is not None:
                opt.set_lower_bounds(self.global_lb)
            if self.global_ub is not None:
                opt.set_upper_bounds(self.global_ub)
            opt.set_xtol_rel(self.xtol_rel)
            opt.set_xtol_abs(self.xtol_abs)
            opt.set_ftol_rel(self.ftol_rel)
            opt.set_ftol_abs(self.ftol_abs)
            opt.set_maxeval(self.maxeval)
            opt.set_maxtime(self.maxtime)
            # couple local optimizer if required
            if local_opt is not None:
                local_opt.set_xtol_rel(self.xtol_rel)
                local_opt.set_xtol_abs(self.xtol_abs)
                local_opt.set_ftol_rel(self.ftol_rel)
                local_opt.set_ftol_abs(self.ftol_abs)
                local_opt.set_maxeval(self.maxeval)
                local_opt.set_maxtime(self.maxtime)
                opt.set_local_optimizer(local_opt)
            if self.objective_func is not None:
                opt.set_min_objective(self.objective_func)
            self._opt = opt
            if self.constraints is not None:
                self._add_constraints()
        return self._opt

    def minimize(self, x0=None):
        """
        Minimize a function using NLOPT
        """
        if x0 is None:
            x0 = self.x0
        else:
            assert len(x0) == self.N
        try:
            x = self.opt.optimize(x0)
        except Exception:
            x = x0
        res = NLOptOptimizationResult(x, self.opt, x0)
        # Reset the optimizer so that it can be reused for a new optimization
        del self._opt
        return res


# TODO: can the optimizer be reused or not?
# TODO: in that case we should do something about it


class NLOptOptimizationResult:
    """
    A wrapper around the result of an NLOPT optimization.

    See documentation for interpretation of return values
    https://nlopt.readthedocs.io/en/latest/NLopt_Reference/#successful-termination-positive-return-values

    For convenience, the result can be converted to Scipy format.
    """

    def __init__(self, x, nlopt_optimizer, x0):
        self._nlopt_optimizer = nlopt_optimizer
        self._x = x
        self._x0 = x0

    @property
    def success(self):
        return self._nlopt_optimizer.last_optimize_result() > 0

    @property
    def _message_code(self):
        return self._nlopt_optimizer.last_optimize_result()

    @property
    def message(self):
        msg_text = {
            1: "NLOPT_SUCCESS, generic success return value.",
            2: "NLOPT_STOPVAL_REACHED, optimization stopped because stopval (above) was reached.",
            3: "NLOPT_FTOL_REACHED, optimization stopped because ftol_rel or ftol_abs (above) was reached.",
            4: "NLOPT_XTOL_REACHED, optimization stopped because xtol_rel or xtol_abs (above) was reached.",
            5: "NLOPT_MAXEVAL_REACHED, optimization stopped because maxeval (above) was reached.",
            6: "NLOPT_MAXTIME_REACHED, optimization stopped because maxtime (above) was reached.",
            -1: "NLOPT_FAILURE, generic failure code.",
            -2: "NLOPT_INVALID_ARGS, invalid arguments (e.g. lower bounds are bigger than upper bounds, an unknown algorithm was specified, etcetera).",
            -3: "NLOPT_OUT_OF_MEMORY, ran out of memory.",
            -4: "NLOPT_ROUNDOFF_LIMITED: Halted because roundoff errors limited progress. (In this case, the optimization still typically returns a useful result.",
            -5: "NLOPT_FORCED_STOP, halted because of a forced termination: the user called nlopt_force_stop(opt) on the optimization’s nlopt_opt object opt from the user’s objective function or constraints.",
        }
        return msg_text[self._message_code]

    @property
    def fx(self):
        """
        The final value of the objective function.
        """
        if not hasattr(self, "_fx"):
            self._fx = self._nlopt_optimizer.last_optimum_value()
        return self._fx

    @property
    def x0(self):
        return self._x0

    @property
    def x(self):
        """
        The final minimizer of func.
        """
        return self._x

    @property
    def its(self):
        """
        The number of function evaluations.
        """
        if not hasattr(self, "_its"):
            self._its = self._nlopt_optimizer.get_numevals()
        return self._its

    @property
    def scipy_result(self):
        """
        Returns a dictionary in scipy format, i.e., one with keys:

        out : ndarray of float
            The final minimizer of func.
        fx : ndarray of float, if full_output is true
            The final value of the objective function.
        its : int, if full_output is true
            The number of function evaluations.
        imode : int, if full_output is true
            The exit mode from the optimizer.
        smode : string, if full_output is true
            Message describing the exit mode from the optimizer

        Note that the imode interpretation differs from scipy.

        See also: https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.optimize.fmin_slsqp.html
        """
        sol = {
            "out": self.x,
            "fx": self.fx,
            "its": self.its,
            "imode": self._message_code,
            "smode": self.message,
        }
        sol["x"] = self.optim_value
        sol["status"] = "optimal" if self.success else self.message
        sol["solver_backend"] = "nlopt"
        return sol
