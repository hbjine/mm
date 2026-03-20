"""
Nonlinear Equation Solver Module
=================================

This module provides classes and functions for solving nonlinear equations
and systems, including root finding and fixed-point iteration.

Classes:
    NonLinearSolver: General-purpose nonlinear equation solver

Functions:
    root: Find roots of a scalar nonlinear equation
    fsolve: Find roots of a system of nonlinear equations
    brentq: Find root using Brent's method (scalar equations)
    fixed_point: Fixed-point iteration for solving equations
"""

import numpy as np
from typing import Callable, Tuple, Union, Optional, Dict, Any

ArrayLike = Union[np.ndarray, list, tuple]
ScalarFunction = Callable[[float, Any], float]
VectorFunction = Callable[[np.ndarray, Any], np.ndarray]

class NonLinearSolver:
    """
    A class for solving nonlinear equations and systems.
    
    Methods:
        bisection: Bisection method (scalar equations)
        newton: Newton-Raphson method (scalar or vector)
        secant: Secant method (scalar equations)
        brentq: Brent's method (scalar equations)
        broyden: Broyden's method (systems of equations)
        fixed_point: Fixed-point iteration
    """
    
    def __init__(self, func: Union[ScalarFunction, VectorFunction], 
                 jac: Optional[Union[Callable, np.ndarray]] = None):
        """
        Initialize the NonLinearSolver.
        
        Args:
            func: The function to find roots for
            jac: Jacobian function or matrix (optional)
        """
        self.func = func
        self.jac = jac
    
    def bisection(self, a: float, b: float, tol: float = 1e-10, max_iter: int = 100, *args) -> float:
        """
        Find root using bisection method.
        
        Args:
            a: Left endpoint of interval
            b: Right endpoint of interval
            tol: Tolerance for convergence
            max_iter: Maximum number of iterations
            *args: Additional arguments to pass to the function
        
        Returns:
            Approximate root
        """
        fa = self.func(a, *args)
        fb = self.func(b, *args)
        
        if fa * fb >= 0:
            raise ValueError("Function has same sign at both endpoints")
        
        for _ in range(max_iter):
            c = (a + b) / 2
            fc = self.func(c, *args)
            
            if abs(fc) < tol or (b - a) / 2 < tol:
                return c
            
            if fa * fc < 0:
                b = c
                fb = fc
            else:
                a = c
                fa = fc
        
        raise ValueError("Bisection method did not converge")
    
    def newton(self, x0: Union[float, ArrayLike], tol: float = 1e-10, max_iter: int = 100, *args) -> Union[float, np.ndarray]:
        """
        Find root using Newton-Raphson method.
        
        Args:
            x0: Initial guess
            tol: Tolerance for convergence
            max_iter: Maximum number of iterations
            *args: Additional arguments to pass to the function
        
        Returns:
            Approximate root
        """
        x = np.asarray(x0, dtype=np.float64)
        is_scalar = x.ndim == 0 or (x.ndim == 1 and len(x) == 1)
        
        for _ in range(max_iter):
            f_val = self.func(x, *args)
            
            if self.jac is not None:
                if callable(self.jac):
                    jac_val = self.jac(x, *args)
                else:
                    jac_val = np.asarray(self.jac, dtype=np.float64)
            else:
                # Numerical Jacobian approximation
                if is_scalar:
                    eps = 1e-8
                    jac_val = (self.func(x + eps, *args) - f_val) / eps
                else:
                    n = len(x)
                    jac_val = np.zeros((n, n))
                    eps = 1e-8
                    for j in range(n):
                        x_pert = x.copy()
                        x_pert[j] += eps
                        jac_val[:, j] = (self.func(x_pert, *args) - f_val) / eps
            
            if is_scalar:
                delta = f_val / jac_val if jac_val != 0 else 0
            else:
                try:
                    delta = np.linalg.solve(jac_val, f_val)
                except np.linalg.LinAlgError:
                    delta = np.linalg.pinv(jac_val) @ f_val
            
            x = x - delta
            
            if np.linalg.norm(f_val) < tol:
                return float(x) if is_scalar else x
        
        raise ValueError("Newton-Raphson method did not converge")
    
    def secant(self, x0: float, x1: float, tol: float = 1e-10, max_iter: int = 100, *args) -> float:
        """
        Find root using secant method.
        
        Args:
            x0: First initial guess
            x1: Second initial guess
            tol: Tolerance for convergence
            max_iter: Maximum number of iterations
            *args: Additional arguments to pass to the function
        
        Returns:
            Approximate root
        """
        f0 = self.func(x0, *args)
        f1 = self.func(x1, *args)
        
        for _ in range(max_iter):
            if abs(f1 - f0) < 1e-15:
                raise ValueError("Division by zero in secant method")
            
            x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
            f2 = self.func(x2, *args)
            
            if abs(f2) < tol or abs(x2 - x1) < tol:
                return x2
            
            x0, x1 = x1, x2
            f0, f1 = f1, f2
        
        raise ValueError("Secant method did not converge")
    
    def brentq(self, a: float, b: float, tol: float = 1e-10, max_iter: int = 100, *args) -> float:
        """
        Find root using Brent's method.
        
        Args:
            a: Left endpoint of interval
            b: Right endpoint of interval
            tol: Tolerance for convergence
            max_iter: Maximum number of iterations
            *args: Additional arguments to pass to the function
        
        Returns:
            Approximate root
        """
        fa = self.func(a, *args)
        fb = self.func(b, *args)
        
        if fa * fb >= 0:
            raise ValueError("Function has same sign at both endpoints")
        
        if abs(fa) < abs(fb):
            a, b = b, a
            fa, fb = fb, fa
        
        c = a
        fc = fa
        d = b - a
        e = d
        
        for _ in range(max_iter):
            if abs(b - a) < tol:
                return b
            
            if abs(fc) < abs(fb):
                a, b, c = b, c, b
                fa, fb, fc = fb, fc, fb
            
            tol1 = 2 * np.finfo(float).eps * abs(b) + 0.5 * tol
            m = 0.5 * (c - b)
            
            if abs(m) < tol1 or fb == 0:
                return b
            
            if abs(e) >= tol1 and abs(fa) > abs(fb):
                s = fb / fa
                if a == c:
                    p = 2 * m * s
                    q = 1 - s
                else:
                    q = fa / fc
                    r = fb / fc
                    p = s * (2 * m * q * (q - r) - (b - a) * (r - 1))
                    q = (q - 1) * (r - 1) * (s - 1)
                
                if p > 0:
                    q = -q
                p = abs(p)
                
                if 2 * p < min(3 * m * q - abs(tol1 * q), abs(e * q)):
                    e = d
                    d = p / q
                else:
                    d = m
                    e = m
            else:
                d = m
                e = m
            
            a = b
            fa = fb
            
            if abs(d) > tol1:
                b += d
            else:
                b += tol1 if m > 0 else -tol1
            
            fb = self.func(b, *args)
            
            if fb * fc > 0:
                c = a
                fc = fa
                d = b - a
                e = d
        
        return b
    
    def broyden(self, x0: ArrayLike, tol: float = 1e-10, max_iter: int = 100, *args) -> np.ndarray:
        """
        Find root using Broyden's method (for systems of equations).
        
        Args:
            x0: Initial guess vector
            tol: Tolerance for convergence
            max_iter: Maximum number of iterations
            *args: Additional arguments to pass to the function
        
        Returns:
            Approximate root vector
        """
        x = np.asarray(x0, dtype=np.float64)
        n = len(x)
        
        # Initial Jacobian approximation (finite differences)
        f0 = self.func(x, *args)
        B = np.zeros((n, n))
        eps = 1e-8
        
        for j in range(n):
            x_pert = x.copy()
            x_pert[j] += eps
            B[:, j] = (self.func(x_pert, *args) - f0) / eps
        
        for _ in range(max_iter):
            try:
                delta = np.linalg.solve(B, -f0)
            except np.linalg.LinAlgError:
                delta = np.linalg.pinv(B) @ (-f0)
            
            x_new = x + delta
            f_new = self.func(x_new, *args)
            y = f_new - f0
            
            # Update Jacobian approximation
            delta_reshaped = delta.reshape(-1, 1)
            y_reshaped = y.reshape(-1, 1)
            B += (y_reshaped - B @ delta_reshaped) @ delta_reshaped.T / (delta_reshaped.T @ delta_reshaped)
            
            if np.linalg.norm(f_new) < tol or np.linalg.norm(delta) < tol:
                return x_new
            
            x = x_new
            f0 = f_new
        
        raise ValueError("Broyden's method did not converge")
    
    def fixed_point(self, x0: Union[float, ArrayLike], g: Optional[Callable] = None, 
                    tol: float = 1e-10, max_iter: int = 100, *args) -> Union[float, np.ndarray]:
        """
        Find fixed point using fixed-point iteration.
        
        Args:
            x0: Initial guess
            g: Fixed-point iteration function (x = g(x))
               If None, uses g(x) = x - func(x)
            tol: Tolerance for convergence
            max_iter: Maximum number of iterations
            *args: Additional arguments to pass to the function
        
        Returns:
            Approximate fixed point
        """
        x = np.asarray(x0, dtype=np.float64)
        is_scalar = x.ndim == 0 or (x.ndim == 1 and len(x) == 1)
        
        if g is None:
            def g(x_val, *g_args):
                return x_val - self.func(x_val, *g_args)
        
        for _ in range(max_iter):
            x_new = g(x, *args)
            
            if np.linalg.norm(x_new - x) < tol:
                return float(x_new) if is_scalar else x_new
            
            x = x_new
        
        raise ValueError("Fixed-point iteration did not converge")

# Core interface functions
def root(fun: ScalarFunction, x0: Union[float, Tuple[float, float]], 
         method: str = 'brentq', **kwargs) -> Dict[str, Any]:
    """
    Find roots of a scalar nonlinear equation.
    
    Args:
        fun: The function to find roots for
        x0: Initial guess or interval [a, b]
        method: Root finding method ('brentq', 'bisect', 'newton', 'secant')
        **kwargs: Additional arguments for the specific method
    
    Returns:
        Dictionary containing:
            root: Approximate root
            success: Boolean indicating success
            message: Status message
    """
    solver = NonLinearSolver(fun)
    
    try:
        if method.lower() == 'brentq':
            if isinstance(x0, (list, tuple)) and len(x0) == 2:
                result = solver.brentq(x0[0], x0[1], **kwargs)
            else:
                raise ValueError("brentq requires an interval [a, b] as x0")
        elif method.lower() == 'bisect':
            if isinstance(x0, (list, tuple)) and len(x0) == 2:
                result = solver.bisection(x0[0], x0[1], **kwargs)
            else:
                raise ValueError("bisect requires an interval [a, b] as x0")
        elif method.lower() == 'newton':
            result = solver.newton(x0, **kwargs)
        elif method.lower() == 'secant':
            if isinstance(x0, (list, tuple)) and len(x0) == 2:
                result = solver.secant(x0[0], x0[1], **kwargs)
            else:
                raise ValueError("secant requires two initial guesses as x0")
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return {
            'root': result,
            'success': True,
            'message': 'Root finding successful.'
        }
    except Exception as e:
        return {
            'root': None,
            'success': False,
            'message': f'Root finding failed: {str(e)}'
        }

def fsolve(fun: VectorFunction, x0: ArrayLike, 
           method: str = 'broyden', **kwargs) -> Dict[str, Any]:
    """
    Find roots of a system of nonlinear equations.
    
    Args:
        fun: The function representing the system of equations
        x0: Initial guess vector
        method: Solver method ('broyden', 'newton')
        **kwargs: Additional arguments for the specific method
    
    Returns:
        Dictionary containing:
            x: Approximate solution vector
            success: Boolean indicating success
            message: Status message
    """
    solver = NonLinearSolver(fun)
    
    try:
        if method.lower() == 'broyden':
            result = solver.broyden(x0, **kwargs)
        elif method.lower() == 'newton':
            result = solver.newton(x0, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return {
            'x': result,
            'success': True,
            'message': 'System solving successful.'
        }
    except Exception as e:
        return {
            'x': None,
            'success': False,
            'message': f'System solving failed: {str(e)}'
        }

def brentq(fun: ScalarFunction, a: float, b: float, **kwargs) -> float:
    """
    Find root using Brent's method.
    
    Args:
        fun: The function to find roots for
        a: Left endpoint of interval
        b: Right endpoint of interval
        **kwargs: Additional arguments
    
    Returns:
        Approximate root
    """
    result = root(fun, [a, b], method='brentq', **kwargs)
    if not result['success']:
        raise ValueError(result['message'])
    return result['root']

def fixed_point(fun: Callable, x0: Union[float, ArrayLike], **kwargs) -> Union[float, np.ndarray]:
    """
    Fixed-point iteration for solving equations.
    
    Args:
        fun: The fixed-point iteration function (x = fun(x))
        x0: Initial guess
        **kwargs: Additional arguments
    
    Returns:
        Approximate fixed point
    """
    solver = NonLinearSolver(lambda x: x - fun(x))
    return solver.fixed_point(x0, g=fun, **kwargs)
