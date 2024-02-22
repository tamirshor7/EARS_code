"""
A collection of helper functions for optimization with JAX.
https://gist.github.com/slinderman/24552af1bdbb6cb033bfea9b2dc4ecfd
scipy's original doc:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
l-bfgs-b options: 
https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html#optimize-minimize-lbfgsb
"""
import numpy as onp
import jax.numpy as jnp
import scipy.optimize
from jax import value_and_grad
from jax.flatten_util import ravel_pytree
import os
from time import time
from jax.config import config
config.update("jax_enable_x64", True)

from torch.utils.tensorboard import SummaryWriter

def minimize(fun, x0,
             method=None,
             args=(),
             bounds=None, 
             constraints=(), 
             tol=None,
             callback=None, 
             options=None,
             exp_name="",
             opt_harmonies_list=[],
             idx_block_opt_params=-1,
             get_opt_params=None,
             step_log_signals=1,
             borders_signals=[1000,2000]):
    """
    A simple wrapper for scipy.optimize.minimize using JAX.
    
    Args: 
        fun: The objective function to be minimized, written in JAX code
        so that it is automatically differentiable.  It is of type,
            ```fun: x, *args -> float```
        where `x` is a PyTree and args is a tuple of the fixed parameters needed 
        to completely specify the function.  
            
        x0: Initial guess represented as a JAX PyTree.
            
        args: tuple, optional. Extra arguments passed to the objective function 
        and its derivative.  Must consist of valid JAX types; e.g. the leaves
        of the PyTree must be floats.
        
        _The remainder of the keyword arguments are inherited from 
        `scipy.optimize.minimize`, and their descriptions are copied here for
        convenience._
        
        method : str or callable, optional
        Type of solver.  Should be one of
            - 'Nelder-Mead' :ref:`(see here) <optimize.minimize-neldermead>`
            - 'Powell'      :ref:`(see here) <optimize.minimize-powell>`
            - 'CG'          :ref:`(see here) <optimize.minimize-cg>`
            - 'BFGS'        :ref:`(see here) <optimize.minimize-bfgs>`
            - 'Newton-CG'   :ref:`(see here) <optimize.minimize-newtoncg>`
            - 'L-BFGS-B'    :ref:`(see here) <optimize.minimize-lbfgsb>`
            - 'TNC'         :ref:`(see here) <optimize.minimize-tnc>`
            - 'COBYLA'      :ref:`(see here) <optimize.minimize-cobyla>`
            - 'SLSQP'       :ref:`(see here) <optimize.minimize-slsqp>`
            - 'trust-constr':ref:`(see here) <optimize.minimize-trustconstr>`
            - 'dogleg'      :ref:`(see here) <optimize.minimize-dogleg>`
            - 'trust-ncg'   :ref:`(see here) <optimize.minimize-trustncg>`
            - 'trust-exact' :ref:`(see here) <optimize.minimize-trustexact>`
            - 'trust-krylov' :ref:`(see here) <optimize.minimize-trustkrylov>`
            - custom - a callable object (added in version 0.14.0),
              see below for description.
        If not given, chosen to be one of ``BFGS``, ``L-BFGS-B``, ``SLSQP``,
        depending if the problem has constraints or bounds.
                
        bounds : sequence or `Bounds`, optional
            Bounds on variables for L-BFGS-B, TNC, SLSQP, Powell, and
            trust-constr methods. There are two ways to specify the bounds:
                1. Instance of `Bounds` class.
                2. Sequence of ``(min, max)`` pairs for each element in `x`. None
                is used to specify no bound.
            Note that in order to use `bounds` you will need to manually flatten
            them in the same order as your inputs `x0`.
            
        constraints : {Constraint, dict} or List of {Constraint, dict}, optional
            Constraints definition (only for COBYLA, SLSQP and trust-constr).
            Constraints for 'trust-constr' are defined as a single object or a
            list of objects specifying constraints to the optimization problem.
            Available constraints are:
                - `LinearConstraint`
                - `NonlinearConstraint`
            Constraints for COBYLA, SLSQP are defined as a list of dictionaries.
            Each dictionary with fields:
                type : str
                    Constraint type: 'eq' for equality, 'ineq' for inequality.
                fun : callable
                    The function defining the constraint.
                jac : callable, optional
                    The Jacobian of `fun` (only for SLSQP).
                args : sequence, optional
                    Extra arguments to be passed to the function and Jacobian.
            Equality constraint means that the constraint function result is to
            be zero whereas inequality means that it is to be non-negative.
            Note that COBYLA only supports inequality constraints.
            
            Note that in order to use `constraints` you will need to manually flatten
            them in the same order as your inputs `x0`.
            
        tol : float, optional
            Tolerance for termination. For detailed control, use solver-specific
            options.
            
        options : dict, optional
            A dictionary of solver options. All methods accept the following
            generic options:
                maxiter : int
                    Maximum number of iterations to perform. Depending on the
                    method each iteration may use several function evaluations.
                disp : bool
                    Set to True to print convergence messages.
            For method-specific options, see :func:`show_options()`.
            
        callback : callable, optional
            Called after each iteration. For 'trust-constr' it is a callable with
            the signature:
                ``callback(xk, OptimizeResult state) -> bool``
            where ``xk`` is the current parameter vector represented as a PyTree,
             and ``state`` is an `OptimizeResult` object, with the same fields
            as the ones from the return. If callback returns True the algorithm 
            execution is terminated.
            
            For all the other methods, the signature is:
                ```callback(xk)```
            where `xk` is the current parameter vector, represented as a PyTree.
            
    Returns:
        res : The optimization result represented as a ``OptimizeResult`` object.
        Important attributes are: 
            ``x``: the solution array, represented as a JAX PyTree 
            ``success``: a Boolean flag indicating if the optimizer exited successfully
            ``message``: describes the cause of the termination. 
        See `scipy.optimize.OptimizeResult` for a description of other attributes.
        
    """
    # Use tree flatten and unflatten to convert params x0 from PyTrees to flat arrays
    x0_flat, unravel = ravel_pytree(x0)
    
    # logging with tfboard
    writer = SummaryWriter(comment=f'_{exp_name}')
    opt_step = 0
    cur_grad = None

    # Wrap the objective function to consume flat _original_ 
    # numpy arrays and produce scalar outputs.
    #fun_jit = jit(fun, static_argnums=(2,3,4,8,9,10,13,14))
    def fun_wrapper(x_flat, *args):
        nonlocal opt_step
        nonlocal cur_grad

        print(f'Processing iteration {opt_step}')
        x = unravel(x_flat)
        #res = fun_jit(x, *(args+(True,)))
        #res = fun(x, *(args+(True,)))
        # loss = float(res[0])
        
        start_time = time()
        loss, cur_grad = value_and_grad(fun)(x, *(args))
        print(f'val and grad time: {time()-start_time}')

        if opt_step % step_log_signals == 0:
            log_params_and_loss_tensorboard(x, loss)

        print(f'Iter: {opt_step}, Loss: {loss}')
        # updating the step number
        opt_step += 1   
        return loss

    # Wrap the gradient in a similar manner
    # grad_jit = jit(grad(fun), static_argnums=(2,3,4,8,9,10,13))
    #grad_f = grad(fun)
    #jax.jit(lambda tree: ravel_pytree(tree)[0])
    def jac_wrapper(x_flat, *args):
        nonlocal cur_grad
        # x = unravel(x_flat)
        # g_res = grad_f(x, *args)
        # g_flat, _ = ravel_pytree(g_res)
        g_flat, _ = ravel_pytree(cur_grad)
        return onp.array(g_flat)
    
    # Wrap the callback to consume a pytree
    def callback_wrapper(x_flat, *args):
        if callback is not None:
            x = unravel(x_flat)
            return callback(x, *args)
    
    # log with TensorBoard <3
    def log_params_and_loss_tensorboard(cur_opt_params, cur_loss):
        nonlocal opt_step
        # log opt params
        _, phies_0_list, magnitudes_list = get_opt_params(cur_opt_params, idx_block_opt_params)

        for radius_idx in range(magnitudes_list.shape[1]):
            for source_idx in range(magnitudes_list.shape[0]):
                for harmony_idx, harmony in enumerate(opt_harmonies_list):
                    writer.add_scalar(f'Signals/Magnitudes/Radius_{radius_idx}/Source_{source_idx}/m_i_{harmony}', 
                                        float(magnitudes_list[source_idx, radius_idx, harmony_idx]), opt_step)
                    writer.add_scalar(f'Signals/Phies/Radius_{radius_idx}/Source_{source_idx}/p_i_{harmony}', 
                                        float(phies_0_list[source_idx, radius_idx, harmony_idx]), opt_step)
        # log loss
        writer.add_scalar("Loss", float(cur_loss), opt_step)

    # Minimize with scipy
    results = scipy.optimize.minimize(fun_wrapper, 
                                      x0_flat, 
                                      args=args,
                                      method=method,
                                      jac=jac_wrapper, 
                                      callback=callback_wrapper, 
                                      bounds=bounds, 
                                      constraints=constraints, 
                                      tol=tol,
                                      options=options)
    
    # pack the output back into a PyTree
    results["x"] = unravel(results["x"])
    return results


# test with rosenbrock function and bfgs
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES']= '3'
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='.20'

    def rosen(x, b, c):
        if c:
            return (x[0,0]-1)**2 + b*(x[0,1]-x[0,0]**2)**2
        else:
            print('Just checking')
            return (x[0]-1)**2 + b*(x[1]-x[0]**2)**2

    x0 = jnp.array([[1.3, 0.7],[4., 5]])
    res = minimize(rosen, x0, args=(10, True), method='BFGS',options={'gtol': 1e-6, 'disp': True, 'maxiter':50})
    print(res)

    # with constraints
    # fun = lambda x: (x[0] - 1)**2 + (x[1] - 2.5)**2
    # cons = ({'type': 'ineq', 'fun': lambda x:  x[0] - 2 * x[1] + 2},
    #      {'type': 'ineq', 'fun': lambda x: -x[0] - 2 * x[1] + 6},
    #      {'type': 'ineq', 'fun': lambda x: -x[0] + 2 * x[1] + 2})
    # bnds = ((0, None), (0, None))
    # res = minimize(fun, (2., 0.), method='SLSQP', bounds=bnds,
    #             constraints=cons)