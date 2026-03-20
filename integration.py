"""
Numerical Integration Module
============================

This module provides classes and functions for numerical integration,
including definite integrals (single and multiple), adaptive integration,
Gaussian quadrature, and discrete data integration.

Classes:
    Integrator: General-purpose numerical integrator with various methods

Functions:
    quad: Compute definite integral of a function over [a, b]
    dblquad: Compute double integral over a rectangular region
    tplquad: Compute triple integral over a rectangular prism region
    trapz: Integrate discrete data using trapezoidal rule
    simps: Integrate discrete data using Simpson's rule
    romberg: Compute definite integral using Romberg integration
    quadgk: Compute definite integral using adaptive Gauss-Kronrod quadrature
    fixed_quad: Compute definite integral using fixed-order Gaussian quadrature
"""

import numpy as np
from typing import Callable, Tuple, Union, Optional

ArrayLike = Union[np.ndarray, list, tuple]
FunctionType = Callable[[float], float]

class Integrator:
    """
    A class for numerical integration with various methods.
    
    Methods:
        trapezoidal: Trapezoidal rule
        simpson: Simpson's rule
        romberg: Romberg integration
        gauss_legendre: Gauss-Legendre quadrature
        adaptive: Adaptive quadrature
    """
    
    def __init__(self, func: FunctionType):
        """
        Initialize the Integrator with the function to integrate.
        
        Args:
            func: The function to integrate
        """
        self.func = np.vectorize(func)
    
    def trapezoidal(self, a: float, b: float, n: int = 100) -> float:
        """
        Compute definite integral using trapezoidal rule.
        
        Args:
            a: Lower limit of integration
            b: Upper limit of integration
            n: Number of subintervals
        
        Returns:
            Approximate integral value
        """
        x = np.linspace(a, b, n + 1)
        y = self.func(x)
        h = (b - a) / n
        return h * (np.sum(y) - 0.5 * (y[0] + y[-1]))
    
    def simpson(self, a: float, b: float, n: int = 100) -> float:
        """
        Compute definite integral using Simpson's rule.
        
        Args:
            a: Lower limit of integration
            b: Upper limit of integration
            n: Number of subintervals (must be even)
        
        Returns:
            Approximate integral value
        """
        if n % 2 != 0:
            n += 1
        
        x = np.linspace(a, b, n + 1)
        y = self.func(x)
        h = (b - a) / n
        
        return h / 3 * (y[0] + y[-1] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-2:2]))
    
    def romberg(self, a: float, b: float, tol: float = 1e-10, max_iter: int = 20) -> float:
        """
        Compute definite integral using Romberg integration.
        
        Args:
            a: Lower limit of integration
            b: Upper limit of integration
            tol: Tolerance for convergence
            max_iter: Maximum number of iterations
        
        Returns:
            Approximate integral value
        """
        R = np.zeros((max_iter, max_iter))
        h = b - a
        
        R[0, 0] = 0.5 * h * (self.func(a) + self.func(b))
        
        for i in range(1, max_iter):
            h /= 2
            sum_val = np.sum(self.func(a + h * np.arange(1, 2**i, 2)))
            R[i, 0] = 0.5 * R[i-1, 0] + h * sum_val
            
            for j in range(1, i + 1):
                R[i, j] = R[i, j-1] + (R[i, j-1] - R[i-1, j-1]) / (4**j - 1)
            
            if i > 1 and abs(R[i, i] - R[i-1, i-1]) < tol:
                return R[i, i]
        
        return R[max_iter-1, max_iter-1]
    
    def gauss_legendre(self, a: float, b: float, n: int = 10) -> float:
        """
        Compute definite integral using Gauss-Legendre quadrature.
        
        Args:
            a: Lower limit of integration
            b: Upper limit of integration
            n: Number of quadrature points
        
        Returns:
            Approximate integral value
        """
        x, w = np.polynomial.legendre.leggauss(n)
        x_transformed = 0.5 * (b - a) * x + 0.5 * (a + b)
        w_transformed = 0.5 * (b - a) * w
        return np.sum(w_transformed * self.func(x_transformed))
    
    def adaptive(self, a: float, b: float, tol: float = 1e-10) -> float:
        """
        Compute definite integral using adaptive quadrature.
        
        Args:
            a: Lower limit of integration
            b: Upper limit of integration
            tol: Tolerance for convergence
        
        Returns:
            Approximate integral value
        """
        def _adaptive(a: float, b: float, fa: float, fb: float, fc: float, S: float) -> float:
            c = 0.5 * (a + b)
            h = b - a
            d = 0.5 * (a + c)
            e = 0.5 * (c + b)
            fd = self.func(d)
            fe = self.func(e)
            
            Sleft = h / 12 * (fa + 4 * fd + fc)
            Sright = h / 12 * (fc + 4 * fe + fb)
            S2 = Sleft + Sright
            
            if abs(S2 - S) <= 15 * tol:
                return S2 + (S2 - S) / 15
            
            return _adaptive(a, c, fa, fc, fd, Sleft) + _adaptive(c, b, fc, fb, fe, Sright)
        
        c = 0.5 * (a + b)
        h = b - a
        fa = self.func(a)
        fb = self.func(b)
        fc = self.func(c)
        S = h / 6 * (fa + 4 * fc + fb)
        
        return _adaptive(a, b, fa, fb, fc, S)

# Core interface functions
def quad(func: FunctionType, a: float, b: float, method: str = 'adaptive', **kwargs) -> Tuple[float, float]:
    """
    Compute definite integral of a function over [a, b].
    
    Args:
        func: The function to integrate
        a: Lower limit of integration
        b: Upper limit of integration
        method: Integration method ('adaptive', 'romberg', 'gauss_legendre', 'trapezoidal', 'simpson')
        **kwargs: Additional arguments for the specific method
    
    Returns:
        integral: Approximate integral value
        error: Estimated error (None for some methods)
    """
    integrator = Integrator(func)
    
    if method == 'adaptive':
        result = integrator.adaptive(a, b, **kwargs)
    elif method == 'romberg':
        result = integrator.romberg(a, b, **kwargs)
    elif method == 'gauss_legendre':
        result = integrator.gauss_legendre(a, b, **kwargs)
    elif method == 'trapezoidal':
        result = integrator.trapezoidal(a, b, **kwargs)
    elif method == 'simpson':
        result = integrator.simpson(a, b, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # For simplicity, return None as error estimate (can be improved)
    return result, None

def dblquad(func: Callable[[float, float], float], a: float, b: float, 
            gfun: Callable[[float], float], hfun: Callable[[float], float]) -> Tuple[float, None]:
    """
    Compute double integral over a region.
    
    Args:
        func: The function to integrate (takes x, y as arguments)
        a: Lower limit of x integration
        b: Upper limit of x integration
        gfun: Function returning lower limit of y for given x
        hfun: Function returning upper limit of y for given x
    
    Returns:
        integral: Approximate integral value
        error: None (placeholder)
    """
    def inner_integral(x: float) -> float:
        y_low = gfun(x)
        y_high = hfun(x)
        return quad(lambda y: func(x, y), y_low, y_high)[0]
    
    result, _ = quad(inner_integral, a, b)
    return result, None

def tplquad(func: Callable[[float, float, float], float], a: float, b: float,
            gfun: Callable[[float], float], hfun: Callable[[float], float],
            qfun: Callable[[float, float], float], rfun: Callable[[float, float], float]) -> Tuple[float, None]:
    """
    Compute triple integral over a region.
    
    Args:
        func: The function to integrate (takes x, y, z as arguments)
        a: Lower limit of x integration
        b: Upper limit of x integration
        gfun: Function returning lower limit of y for given x
        hfun: Function returning upper limit of y for given x
        qfun: Function returning lower limit of z for given x, y
        rfun: Function returning upper limit of z for given x, y
    
    Returns:
        integral: Approximate integral value
        error: None (placeholder)
    """
    def middle_integral(x: float) -> float:
        y_low = gfun(x)
        y_high = hfun(x)
        
        def inner_integral(y: float) -> float:
            z_low = qfun(x, y)
            z_high = rfun(x, y)
            return quad(lambda z: func(x, y, z), z_low, z_high)[0]
        
        return quad(inner_integral, y_low, y_high)[0]
    
    result, _ = quad(middle_integral, a, b)
    return result, None

def trapz(y: ArrayLike, x: Optional[ArrayLike] = None, dx: float = 1.0) -> float:
    """
    Integrate discrete data using trapezoidal rule.
    
    Args:
        y: Array of y-values
        x: Array of x-values (optional, defaults to equally spaced with dx)
        dx: Spacing between x-values (used if x is None)
    
    Returns:
        Approximate integral value
    """
    y = np.asarray(y, dtype=np.float64)
    if x is None:
        return np.trapezoid(y, dx=dx)
    else:
        x = np.asarray(x, dtype=np.float64)
        return np.trapezoid(y, x=x)

def simps(y: ArrayLike, x: Optional[ArrayLike] = None, dx: float = 1.0) -> float:
    """
    Integrate discrete data using Simpson's rule.
    
    Args:
        y: Array of y-values (length must be odd)
        x: Array of x-values (optional, defaults to equally spaced with dx)
        dx: Spacing between x-values (used if x is None)
    
    Returns:
        Approximate integral value
    """
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    
    if n % 2 == 0:
        raise ValueError("Length of y must be odd for Simpson's rule")
    
    if x is None:
        h = dx
    else:
        x = np.asarray(x, dtype=np.float64)
        h = np.diff(x)
    
    if np.isscalar(h):
        return h / 3 * (y[0] + y[-1] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-2:2]))
    else:
        # Non-uniform spacing
        h0 = h[::2]
        h1 = h[1::2]
        h_sum = h0 + h1
        return np.sum(h_sum / 6 * (y[:-2:2] + y[2::2] + 4 * y[1:-1:2]))

# Additional integration functions
def romberg(func: FunctionType, a: float, b: float, **kwargs) -> float:
    """Compute definite integral using Romberg integration."""
    return quad(func, a, b, method='romberg', **kwargs)[0]

def quadgk(func: FunctionType, a: float, b: float, **kwargs) -> Tuple[float, float]:
    """Compute definite integral using adaptive Gauss-Kronrod quadrature."""
    # For simplicity, use adaptive method as a stand-in for Gauss-Kronrod
    return quad(func, a, b, method='adaptive', **kwargs)

def fixed_quad(func: FunctionType, a: float, b: float, n: int = 5) -> Tuple[float, None]:
    """Compute definite integral using fixed-order Gaussian quadrature."""
    return quad(func, a, b, method='gauss_legendre', n=n)
