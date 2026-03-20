"""
Ordinary Differential Equation (ODE) Solver Module
==================================================

This module provides classes and functions for solving ordinary differential
equations, including initial value problems (IVPs) for both stiff and non-stiff
systems.

Classes:
    ODESolver: General-purpose ODE solver with various integration methods

Functions:
    solve_ivp: Solve initial value problem for ODE systems
    ode: Create an ODE solver object (similar to scipy.integrate.ode)
    odeint: Solve ODE using LSODA method (similar to scipy.integrate.odeint)
    lorenz: Compute Lorenz attractor system derivatives
"""

import numpy as np
from typing import Callable, Tuple, Union, Optional, Dict, Any

ArrayLike = Union[np.ndarray, list, tuple]
ODESystem = Callable[[float, np.ndarray, Any], np.ndarray]

class ODESolver:
    """
    A class for solving ordinary differential equations (ODEs).
    
    Methods:
        euler: Euler method (first-order)
        midpoint: Midpoint method (second-order)
        rk4: Runge-Kutta 4th order method
        rk45: Runge-Kutta-Fehlberg method (adaptive step size)
        backward_euler: Backward Euler method (for stiff systems)
        bdf2: Backward Differentiation Formula (second-order, for stiff systems)
    """
    
    def __init__(self, func: ODESystem, jac: Optional[Callable] = None):
        """
        Initialize the ODESolver with the ODE system function.
        
        Args:
            func: The ODE system function (t, y, *args) -> dy/dt
            jac: Jacobian matrix function (t, y, *args) -> df/dy (optional)
        """
        self.func = func
        self.jac = jac
    
    def euler(self, t_span: Tuple[float, float], y0: ArrayLike, h: float = 0.01, *args) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve ODE using Euler method.
        
        Args:
            t_span: Time interval (t0, tf)
            y0: Initial state vector
            h: Step size
            *args: Additional arguments to pass to the ODE function
        
        Returns:
            t: Array of time points
            y: Array of state vectors at each time point
        """
        t0, tf = t_span
        y0 = np.asarray(y0, dtype=np.float64)
        
        t = np.arange(t0, tf + h, h)
        y = np.zeros((len(t), len(y0)))
        y[0] = y0
        
        for i in range(len(t) - 1):
            y[i+1] = y[i] + h * self.func(t[i], y[i], *args)
        
        return t, y
    
    def midpoint(self, t_span: Tuple[float, float], y0: ArrayLike, h: float = 0.01, *args) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve ODE using midpoint method.
        
        Args:
            t_span: Time interval (t0, tf)
            y0: Initial state vector
            h: Step size
            *args: Additional arguments to pass to the ODE function
        
        Returns:
            t: Array of time points
            y: Array of state vectors at each time point
        """
        t0, tf = t_span
        y0 = np.asarray(y0, dtype=np.float64)
        
        t = np.arange(t0, tf + h, h)
        y = np.zeros((len(t), len(y0)))
        y[0] = y0
        
        for i in range(len(t) - 1):
            k1 = self.func(t[i], y[i], *args)
            k2 = self.func(t[i] + h/2, y[i] + h/2 * k1, *args)
            y[i+1] = y[i] + h * k2
        
        return t, y
    
    def rk4(self, t_span: Tuple[float, float], y0: ArrayLike, h: float = 0.01, *args) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve ODE using Runge-Kutta 4th order method.
        
        Args:
            t_span: Time interval (t0, tf)
            y0: Initial state vector
            h: Step size
            *args: Additional arguments to pass to the ODE function
        
        Returns:
            t: Array of time points
            y: Array of state vectors at each time point
        """
        t0, tf = t_span
        y0 = np.asarray(y0, dtype=np.float64)
        
        t = np.arange(t0, tf + h, h)
        y = np.zeros((len(t), len(y0)))
        y[0] = y0
        
        for i in range(len(t) - 1):
            k1 = self.func(t[i], y[i], *args)
            k2 = self.func(t[i] + h/2, y[i] + h/2 * k1, *args)
            k3 = self.func(t[i] + h/2, y[i] + h/2 * k2, *args)
            k4 = self.func(t[i] + h, y[i] + h * k3, *args)
            y[i+1] = y[i] + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        
        return t, y
    
    def rk45(self, t_span: Tuple[float, float], y0: ArrayLike, rtol: float = 1e-6, atol: float = 1e-8, *args) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve ODE using Runge-Kutta-Fehlberg method (adaptive step size).
        
        Args:
            t_span: Time interval (t0, tf)
            y0: Initial state vector
            rtol: Relative tolerance
            atol: Absolute tolerance
            *args: Additional arguments to pass to the ODE function
        
        Returns:
            t: Array of time points
            y: Array of state vectors at each time point
        """
        t0, tf = t_span
        y0 = np.asarray(y0, dtype=np.float64)
        
        # RK45 coefficients
        a = [0, 1/4, 3/8, 12/13, 1, 1/2]
        b = [[], [1/4], [3/32, 9/32], [1932/2197, -7200/2197, 7296/2197],
             [439/216, -8, 3680/513, -845/4104], [-8/27, 2, -3544/2565, 1859/4104, -11/40]]
        c4 = [25/216, 0, 1408/2565, 2197/4104, -1/5, 0]
        c5 = [16/135, 0, 6656/12825, 28561/56430, -9/50, 2/55]
        
        t = [t0]
        y = [y0]
        h = 1e-4  # Initial step size
        
        while t[-1] < tf:
            current_t = t[-1]
            current_y = y[-1]
            
            # Compute k values
            k = []
            for i in range(6):
                ti = current_t + a[i] * h
                yi = current_y + h * sum(b[i][j] * k[j] for j in range(i))
                k.append(self.func(ti, yi, *args))
            
            # Compute error and new step
            y4 = current_y + h * sum(c4[i] * k[i] for i in range(6))
            y5 = current_y + h * sum(c5[i] * k[i] for i in range(6))
            error = np.linalg.norm(y5 - y4)
            
            # Adjust step size
            scale = atol + rtol * np.maximum(np.abs(current_y), np.abs(y5))
            error_norm = np.linalg.norm(error / scale) / np.sqrt(len(current_y))
            
            if error_norm < 1:
                t.append(current_t + h)
                y.append(y5)
                h = min(h * (1 / error_norm) ** 0.25, 4 * h, tf - current_t)
            else:
                h = max(h * (1 / error_norm) ** 0.25, 0.1 * h)
        
        return np.array(t), np.array(y)
    
    def backward_euler(self, t_span: Tuple[float, float], y0: ArrayLike, h: float = 0.01, max_iter: int = 10, *args) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve ODE using backward Euler method (for stiff systems).
        
        Args:
            t_span: Time interval (t0, tf)
            y0: Initial state vector
            h: Step size
            max_iter: Maximum iterations for Newton-Raphson
            *args: Additional arguments to pass to the ODE function
        
        Returns:
            t: Array of time points
            y: Array of state vectors at each time point
        """
        t0, tf = t_span
        y0 = np.asarray(y0, dtype=np.float64)
        n = len(y0)
        
        t = np.arange(t0, tf + h, h)
        y = np.zeros((len(t), n))
        y[0] = y0
        
        I = np.eye(n)
        
        for i in range(len(t) - 1):
            next_t = t[i+1]
            guess = y[i] + h * self.func(t[i], y[i], *args)
            
            for _ in range(max_iter):
                f_val = self.func(next_t, guess, *args)
                if self.jac is not None:
                    jac_val = self.jac(next_t, guess, *args)
                else:
                    # Numerical Jacobian approximation
                    jac_val = np.zeros((n, n))
                    eps = 1e-8
                    for j in range(n):
                        y_pert = guess.copy()
                        y_pert[j] += eps
                        jac_val[:, j] = (self.func(next_t, y_pert, *args) - f_val) / eps
                
                residual = guess - y[i] - h * f_val
                jac_matrix = I - h * jac_val
                delta = np.linalg.solve(jac_matrix, residual)
                guess -= delta
                
                if np.linalg.norm(delta) < 1e-8:
                    break
            
            y[i+1] = guess
        
        return t, y
    
    def bdf2(self, t_span: Tuple[float, float], y0: ArrayLike, h: float = 0.01, max_iter: int = 10, *args) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve ODE using Backward Differentiation Formula (second-order, for stiff systems).
        
        Args:
            t_span: Time interval (t0, tf)
            y0: Initial state vector
            h: Step size
            max_iter: Maximum iterations for Newton-Raphson
            *args: Additional arguments to pass to the ODE function
        
        Returns:
            t: Array of time points
            y: Array of state vectors at each time point
        """
        t0, tf = t_span
        y0 = np.asarray(y0, dtype=np.float64)
        n = len(y0)
        
        t = np.arange(t0, tf + h, h)
        y = np.zeros((len(t), n))
        y[0] = y0
        
        # Use backward Euler for first step
        if len(t) > 1:
            I = np.eye(n)
            next_t = t[1]
            guess = y[0] + h * self.func(t[0], y[0], *args)
            
            for _ in range(max_iter):
                f_val = self.func(next_t, guess, *args)
                residual = guess - y[0] - h * f_val
                
                if self.jac is not None:
                    jac_val = self.jac(next_t, guess, *args)
                else:
                    jac_val = np.zeros((n, n))
                    eps = 1e-8
                    for j in range(n):
                        y_pert = guess.copy()
                        y_pert[j] += eps
                        jac_val[:, j] = (self.func(next_t, y_pert, *args) - f_val) / eps
                
                jac_matrix = I - h * jac_val
                delta = np.linalg.solve(jac_matrix, residual)
                guess -= delta
                
                if np.linalg.norm(delta) < 1e-8:
                    break
            
            y[1] = guess
        
        # BDF2 for subsequent steps
        for i in range(1, len(t) - 1):
            next_t = t[i+1]
            guess = (4 * y[i] - y[i-1]) / 3
            
            for _ in range(max_iter):
                f_val = self.func(next_t, guess, *args)
                residual = 3*guess - 4*y[i] + y[i-1] - 2*h * f_val
                
                if self.jac is not None:
                    jac_val = self.jac(next_t, guess, *args)
                else:
                    jac_val = np.zeros((n, n))
                    eps = 1e-8
                    for j in range(n):
                        y_pert = guess.copy()
                        y_pert[j] += eps
                        jac_val[:, j] = (self.func(next_t, y_pert, *args) - f_val) / eps
                
                jac_matrix = 3*np.eye(n) - 2*h * jac_val
                delta = np.linalg.solve(jac_matrix, residual)
                guess -= delta
                
                if np.linalg.norm(delta) < 1e-8:
                    break
            
            y[i+1] = guess
        
        return t, y

# Core interface functions
def solve_ivp(fun: ODESystem, t_span: Tuple[float, float], y0: ArrayLike, 
              method: str = 'RK45', jac: Optional[Callable] = None, **kwargs) -> Dict[str, Any]:
    """
    Solve initial value problem for ODE systems.
    
    Args:
        fun: The ODE system function (t, y, *args) -> dy/dt
        t_span: Time interval (t0, tf)
        y0: Initial state vector
        method: Integration method ('RK45', 'RK4', 'Euler', 'Midpoint', 'BackwardEuler', 'BDF2')
        jac: Jacobian matrix function (optional)
        **kwargs: Additional arguments for the specific method
    
    Returns:
        Dictionary containing:
            t: Array of time points
            y: Array of state vectors at each time point
            success: Boolean indicating success
            message: Status message
    """
    solver = ODESolver(fun, jac=jac)
    y0 = np.asarray(y0, dtype=np.float64)
    
    try:
        if method.upper() == 'RK45':
            t, y = solver.rk45(t_span, y0, **kwargs)
        elif method.upper() == 'RK4':
            t, y = solver.rk4(t_span, y0, **kwargs)
        elif method.upper() == 'EULER':
            t, y = solver.euler(t_span, y0, **kwargs)
        elif method.upper() == 'MIDPOINT':
            t, y = solver.midpoint(t_span, y0, **kwargs)
        elif method.upper() == 'BACKWARDEULER':
            t, y = solver.backward_euler(t_span, y0, **kwargs)
        elif method.upper() == 'BDF2':
            t, y = solver.bdf2(t_span, y0, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return {
            't': t,
            'y': y.T,  # Transpose to match scipy's shape (n_vars, n_timepoints)
            'success': True,
            'message': 'Integration successful.'
        }
    except Exception as e:
        return {
            't': None,
            'y': None,
            'success': False,
            'message': f'Integration failed: {str(e)}'
        }

def ode(fun: ODESystem, jac: Optional[Callable] = None) -> ODESolver:
    """
    Create an ODE solver object.
    
    Args:
        fun: The ODE system function
        jac: Jacobian matrix function (optional)
    
    Returns:
        ODESolver object
    """
    return ODESolver(fun, jac=jac)

def odeint(fun: ODESystem, y0: ArrayLike, t: ArrayLike, args: tuple = (), **kwargs) -> np.ndarray:
    """
    Solve ODE using LSODA-like method (adaptive step size).
    
    Args:
        fun: The ODE system function (y, t, *args) -> dy/dt
        y0: Initial state vector
        t: Array of time points
        args: Additional arguments to pass to the ODE function
        **kwargs: Additional arguments for the solver
    
    Returns:
        Array of state vectors at each time point
    """
    # Convert function signature from (y, t, ...) to (t, y, ...)
    def fun_t(t, y, *args):
        return fun(y, t, *args)
    
    t_span = (t[0], t[-1])
    result = solve_ivp(fun_t, t_span, y0, method='RK45', args=args, **kwargs)
    
    if not result['success']:
        raise ValueError(result['message'])
    
    # Interpolate to requested time points
    from scipy.interpolate import interp1d
    interp_func = interp1d(result['t'], result['y'], kind='cubic', axis=1)
    return interp_func(t).T

# Example ODE systems
def lorenz(t: float, y: np.ndarray, sigma: float = 10.0, rho: float = 28.0, beta: float = 8/3) -> np.ndarray:
    """
    Compute Lorenz attractor system derivatives.
    
    Args:
        t: Time (not used)
        y: State vector [x, y, z]
        sigma: Lorenz parameter
        rho: Lorenz parameter
        beta: Lorenz parameter
    
    Returns:
        Derivatives [dx/dt, dy/dt, dz/dt]
    """
    x, y, z = y
    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z
    return np.array([dx_dt, dy_dt, dz_dt])

def lorenz_jac(t: float, y: np.ndarray, sigma: float = 10.0, rho: float = 28.0, beta: float = 8/3) -> np.ndarray:
    """
    Compute Jacobian of the Lorenz system.
    
    Args:
        t: Time (not used)
        y: State vector [x, y, z]
        sigma: Lorenz parameter
        rho: Lorenz parameter
        beta: Lorenz parameter
    
    Returns:
        Jacobian matrix
    """
    x, y, z = y
    return np.array([
        [-sigma, sigma, 0],
        [rho - z, -1, -x],
        [y, x, -beta]
    ])
