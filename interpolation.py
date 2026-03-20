"""
Interpolation and Approximation Module
======================================

This module provides classes and functions for interpolating and approximating
data, including 1D and multidimensional interpolation, spline interpolation,
and data smoothing.

Classes:
    Interpolator: General-purpose interpolator with various methods

Functions:
    interp1d: 1D interpolation function
    splrep: Compute B-spline representation of 1D data
    splev: Evaluate B-spline or its derivatives
    griddata: Interpolate unstructured data to a structured grid
    smoothdata: Smooth noisy data using various methods
    interpn: Multidimensional interpolation on regular grids
    pchip: Piecewise Cubic Hermite Interpolating Polynomial
    akima: Akima spline interpolation
"""

import numpy as np
from typing import Callable, Tuple, Union, Optional, List, Dict, Any
from scipy.interpolate import (
    interp1d as scipy_interp1d,
    splrep as scipy_splrep,
    splev as scipy_splev,
    griddata as scipy_griddata,
    RegularGridInterpolator,
    PchipInterpolator,
    Akima1DInterpolator,
    make_interp_spline,
    UnivariateSpline
)
from scipy.ndimage import gaussian_filter1d

ArrayLike = Union[np.ndarray, list, tuple]

class Interpolator:
    """
    A class for data interpolation and approximation.
    
    Methods:
        linear: Linear interpolation
        nearest: Nearest neighbor interpolation
        spline: Spline interpolation (1D)
        cubic: Cubic interpolation
        pchip: Piecewise Cubic Hermite Interpolating Polynomial
        akima: Akima spline interpolation
        smooth: Data smoothing
    """
    
    def __init__(self, x: ArrayLike, y: ArrayLike, kind: str = 'linear', **kwargs):
        """
        Initialize the Interpolator.
        
        Args:
            x: x-values of data points
            y: y-values of data points
            kind: Interpolation method
            **kwargs: Additional arguments for the specific method
        """
        self.x = np.asarray(x, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.float64)
        self.kind = kind.lower()
        self.kwargs = kwargs
        
        if self.x.ndim != 1:
            raise ValueError("x must be 1-dimensional")
        
        if self.y.ndim not in (1, 2):
            raise ValueError("y must be 1 or 2-dimensional")
        
        if self.y.ndim == 1:
            self.y = self.y.reshape(-1, 1)
        
        if len(self.x) != self.y.shape[0]:
            raise ValueError("x and y must have the same length")
        
        # Sort data by x
        sort_idx = np.argsort(self.x)
        self.x = self.x[sort_idx]
        self.y = self.y[sort_idx]
        
        self._interpolator = self._create_interpolator()
    
    def _create_interpolator(self) -> Callable:
        """Create the underlying interpolator object."""
        if self.kind in ('linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic', 'previous', 'next'):
            return scipy_interp1d(self.x, self.y, kind=self.kind, axis=0, **self.kwargs)
        elif self.kind == 'pchip':
            return PchipInterpolator(self.x, self.y, axis=0, **self.kwargs)
        elif self.kind == 'akima':
            return Akima1DInterpolator(self.x, self.y, axis=0, **self.kwargs)
        elif self.kind == 'spline':
            k = self.kwargs.get('k', 3)
            s = self.kwargs.get('s', 0)
            return UnivariateSpline(self.x, self.y, k=k, s=s, **self.kwargs)
        else:
            raise ValueError(f"Unknown interpolation kind: {self.kind}")
    
    def __call__(self, x_new: ArrayLike) -> np.ndarray:
        """
        Evaluate the interpolator at new points.
        
        Args:
            x_new: Points at which to evaluate the interpolator
        
        Returns:
            Interpolated values
        """
        x_new = np.asarray(x_new, dtype=np.float64)
        result = self._interpolator(x_new)
        
        # Handle edge cases for methods that might return NaN
        if self.kind in ('pchip', 'akima'):
            # These methods handle extrapolation differently
            pass
        
        # Squeeze to 1D if only one output dimension
        if result.ndim == 2 and result.shape[1] == 1:
            result = result.ravel()
        
        return result
    
    def derivative(self, x: ArrayLike, order: int = 1) -> np.ndarray:
        """
        Compute the derivative of the interpolator at given points.
        
        Args:
            x: Points at which to evaluate the derivative
            order: Order of derivative
        
        Returns:
            Derivative values
        """
        x = np.asarray(x, dtype=np.float64)
        
        if hasattr(self._interpolator, 'derivative'):
            deriv = self._interpolator.derivative(order)
            return deriv(x)
        else:
            # Finite difference approximation for methods without derivative
            h = np.finfo(float).eps ** (1 / (order + 1))
            return np.gradient(self(x), x, edge_order=2)

# Core interface functions
def interp1d(x: ArrayLike, y: ArrayLike, kind: str = 'linear', **kwargs) -> Callable:
    """
    Create a 1D interpolation function.
    
    Args:
        x: x-values of data points
        y: y-values of data points
        kind: Interpolation method ('linear', 'nearest', 'cubic', 'pchip', 'akima', 'spline')
        **kwargs: Additional arguments
    
    Returns:
        Interpolation function that can be called with new x values
    """
    return Interpolator(x, y, kind=kind, **kwargs)

def splrep(x: ArrayLike, y: ArrayLike, w: Optional[ArrayLike] = None, 
           k: int = 3, s: float = 0, **kwargs) -> Tuple:
    """
    Compute B-spline representation of 1D data.
    
    Args:
        x: x-values of data points
        y: y-values of data points
        w: Weights for each data point (optional)
        k: Degree of spline (1 <= k <= 5)
        s: Smoothing factor
        **kwargs: Additional arguments
    
    Returns:
        Tuple (t, c, k) containing knots, coefficients, and degree
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    
    if w is not None:
        w = np.asarray(w, dtype=np.float64)
    
    return scipy_splrep(x, y, w=w, k=k, s=s, **kwargs)

def splev(x: ArrayLike, tck: Tuple, der: int = 0, ext: int = 0) -> np.ndarray:
    """
    Evaluate B-spline or its derivatives.
    
    Args:
        x: Points at which to evaluate the spline
        tck: Tuple (t, c, k) from splrep
        der: Order of derivative
        ext: Controls extrapolation behavior
    
    Returns:
        Evaluated spline values or derivatives
    """
    x = np.asarray(x, dtype=np.float64)
    return scipy_splev(x, tck, der=der, ext=ext)

def griddata(points: ArrayLike, values: ArrayLike, xi: ArrayLike, 
             method: str = 'linear', **kwargs) -> np.ndarray:
    """
    Interpolate unstructured data to a structured grid.
    
    Args:
        points: Data point coordinates (n_points, n_dims)
        values: Values at data points (n_points,) or (n_points, n_values)
        xi: Grid points at which to interpolate
        method: Interpolation method ('linear', 'nearest', 'cubic')
        **kwargs: Additional arguments
    
    Returns:
        Interpolated values on the grid
    """
    points = np.asarray(points, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    xi = np.asarray(xi, dtype=np.float64)
    
    return scipy_griddata(points, values, xi, method=method, **kwargs)

def smoothdata(y: ArrayLike, method: str = 'gaussian', **kwargs) -> np.ndarray:
    """
    Smooth noisy data using various methods.
    
    Args:
        y: Data to smooth
        method: Smoothing method ('gaussian', 'moving_average', 'spline', 'loess')
        **kwargs: Additional arguments (e.g., sigma for gaussian, window_size for moving average)
    
    Returns:
        Smoothed data
    """
    y = np.asarray(y, dtype=np.float64)
    
    if method.lower() == 'gaussian':
        sigma = kwargs.get('sigma', 1.0)
        return gaussian_filter1d(y, sigma=sigma)
    
    elif method.lower() == 'moving_average':
        window_size = kwargs.get('window_size', 5)
        kernel = np.ones(window_size) / window_size
        return np.convolve(y, kernel, mode='same')
    
    elif method.lower() == 'spline':
        x = kwargs.get('x', np.arange(len(y)))
        s = kwargs.get('s', len(y) * np.var(y) * 0.1)
        k = kwargs.get('k', 3)
        spline = UnivariateSpline(x, y, s=s, k=k)
        return spline(x)
    
    elif method.lower() == 'loess':
        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess
            x = kwargs.get('x', np.arange(len(y)))
            frac = kwargs.get('frac', 0.1)
            result = lowess(y, x, frac=frac)
            return result[:, 1]
        except ImportError:
            raise ImportError("statsmodels is required for LOESS smoothing")
    
    else:
        raise ValueError(f"Unknown smoothing method: {method}")

def interpn(points: List[ArrayLike], values: ArrayLike, xi: ArrayLike, 
            method: str = 'linear', **kwargs) -> np.ndarray:
    """
    Multidimensional interpolation on regular grids.
    
    Args:
        points: List of arrays defining the grid points in each dimension
        values: Values on the regular grid
        xi: Points at which to interpolate
        method: Interpolation method ('linear', 'nearest')
        **kwargs: Additional arguments
    
    Returns:
        Interpolated values
    """
    points = [np.asarray(p, dtype=np.float64) for p in points]
    values = np.asarray(values, dtype=np.float64)
    xi = np.asarray(xi, dtype=np.float64)
    
    interpolator = RegularGridInterpolator(points, values, method=method, **kwargs)
    return interpolator(xi)

def pchip(x: ArrayLike, y: ArrayLike, **kwargs) -> Callable:
    """
    Create a Piecewise Cubic Hermite Interpolating Polynomial (PCHIP).
    
    Args:
        x: x-values of data points
        y: y-values of data points
        **kwargs: Additional arguments
    
    Returns:
        PCHIP interpolation function
    """
    return Interpolator(x, y, kind='pchip', **kwargs)

def akima(x: ArrayLike, y: ArrayLike, **kwargs) -> Callable:
    """
    Create an Akima spline interpolator.
    
    Args:
        x: x-values of data points
        y: y-values of data points
        **kwargs: Additional arguments
    
    Returns:
        Akima spline interpolation function
    """
    return Interpolator(x, y, kind='akima', **kwargs)

def make_spline(x: ArrayLike, y: ArrayLike, k: int = 3, **kwargs) -> Callable:
    """
    Create a smoothing spline interpolator.
    
    Args:
        x: x-values of data points
        y: y-values of data points
        k: Degree of spline
        **kwargs: Additional arguments
    
    Returns:
        Spline interpolation function
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return make_interp_spline(x, y, k=k, **kwargs)
