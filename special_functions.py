"""
Special Functions Module
========================

This module provides implementations of various special functions used in
physics, engineering, and applied mathematics, including gamma functions,
Bessel functions, error functions, and hypergeometric functions.

Classes:
    SpecialFunction: Class providing access to various special functions

Functions:
    gamma: Gamma function
    factorial: Factorial function (integer and real)
    beta: Beta function
    erf: Error function
    erfc: Complementary error function
    besselj: Bessel function of the first kind
    bessely: Bessel function of the second kind
    besseli: Modified Bessel function of the first kind
    besselk: Modified Bessel function of the second kind
    airy: Airy functions
    hypergeom: Generalized hypergeometric function
    legendre: Legendre polynomials
    hermite: Hermite polynomials
    laguerre: Laguerre polynomials
    jacobi: Jacobi polynomials
    zeta: Riemann zeta function
    polygamma: Polygamma function
    digamma: Digamma function (psi function)
    trigamma: Trigamma function
"""

import numpy as np
from scipy.special import (
    gamma as scipy_gamma,
    factorial as scipy_factorial,
    beta as scipy_beta,
    erf as scipy_erf,
    erfc as scipy_erfc,
    jv as scipy_besselj,
    yv as scipy_bessely,
    iv as scipy_besseli,
    kv as scipy_besselk,
    airy as scipy_airy,
    hyp2f1 as scipy_hyp2f1,
    legendre as scipy_legendre,
    hermite as scipy_hermite,
    laguerre as scipy_laguerre,
    jacobi as scipy_jacobi,
    zeta as scipy_zeta,
    polygamma as scipy_polygamma,
    digamma as scipy_digamma,
    gamma,
    loggamma,
    gammainc,
    gammaincc,
    erfcinv,
    erfinv,
    expi,
    expn,
    k0,
    k1,
    kn,
    i0,
    i1,
    j0,
    j1,
    jn,
    y0,
    y1,
    yn
)
from typing import Union, Optional, Tuple, Any

ArrayLike = Union[np.ndarray, list, tuple, float, int]

class SpecialFunction:
    """
    A class providing access to various special functions.
    
    This class serves as a wrapper around scipy.special functions, providing
    a consistent interface and additional documentation.
    
    Methods:
        gamma: Gamma function
        factorial: Factorial function
        beta: Beta function
        erf: Error function
        erfc: Complementary error function
        besselj: Bessel function of the first kind
        bessely: Bessel function of the second kind
        besseli: Modified Bessel function of the first kind
        besselk: Modified Bessel function of the second kind
        airy: Airy functions
        hypergeom: Generalized hypergeometric function
        legendre: Legendre polynomials
        hermite: Hermite polynomials
        laguerre: Laguerre polynomials
        jacobi: Jacobi polynomials
        zeta: Riemann zeta function
        polygamma: Polygamma function
        digamma: Digamma function
        trigamma: Trigamma function
    """
    
    @staticmethod
    def gamma(z: ArrayLike) -> np.ndarray:
        """
        Gamma function.
        
        The gamma function is defined as:
        Γ(z) = ∫₀^∞ t^(z-1) e^(-t) dt
        
        Args:
            z: Input values
        
        Returns:
            Gamma function evaluated at z
        """
        return scipy_gamma(z)
    
    @staticmethod
    def loggamma(z: ArrayLike) -> np.ndarray:
        """
        Natural logarithm of the absolute value of the gamma function.
        
        Args:
            z: Input values
        
        Returns:
            Logarithm of |Γ(z)|
        """
        return loggamma(z)
    
    @staticmethod
    def factorial(n: ArrayLike, exact: bool = False) -> Union[np.ndarray, int]:
        """
        Factorial function.
        
        Args:
            n: Input values (integer or array of integers)
            exact: If True, compute exact integer factorial (for small n)
        
        Returns:
            Factorial of n
        """
        return scipy_factorial(n, exact=exact)
    
    @staticmethod
    def beta(a: ArrayLike, b: ArrayLike) -> np.ndarray:
        """
        Beta function.
        
        The beta function is defined as:
        B(a, b) = ∫₀¹ t^(a-1) (1-t)^(b-1) dt = Γ(a)Γ(b)/Γ(a+b)
        
        Args:
            a: First parameter
            b: Second parameter
        
        Returns:
            Beta function evaluated at (a, b)
        """
        return scipy_beta(a, b)
    
    @staticmethod
    def erf(z: ArrayLike) -> np.ndarray:
        """
        Error function.
        
        The error function is defined as:
        erf(z) = (2/√π) ∫₀^z e^(-t²) dt
        
        Args:
            z: Input values
        
        Returns:
            Error function evaluated at z
        """
        return scipy_erf(z)
    
    @staticmethod
    def erfc(z: ArrayLike) -> np.ndarray:
        """
        Complementary error function.
        
        The complementary error function is defined as:
        erfc(z) = 1 - erf(z)
        
        Args:
            z: Input values
        
        Returns:
            Complementary error function evaluated at z
        """
        return scipy_erfc(z)
    
    @staticmethod
    def erfinv(y: ArrayLike) -> np.ndarray:
        """
        Inverse of the error function.
        
        Args:
            y: Input values (must be between -1 and 1)
        
        Returns:
            Inverse error function evaluated at y
        """
        return erfinv(y)
    
    @staticmethod
    def erfcinv(y: ArrayLike) -> np.ndarray:
        """
        Inverse of the complementary error function.
        
        Args:
            y: Input values (must be between 0 and 2)
        
        Returns:
            Inverse complementary error function evaluated at y
        """
        return erfcinv(y)
    
    @staticmethod
    def besselj(v: ArrayLike, z: ArrayLike) -> np.ndarray:
        """
        Bessel function of the first kind of order v.
        
        Bessel functions satisfy the differential equation:
        z² y'' + z y' + (z² - v²) y = 0
        
        Args:
            v: Order of the Bessel function
            z: Input values
        
        Returns:
            Bessel function of the first kind evaluated at z
        """
        return scipy_besselj(v, z)
    
    @staticmethod
    def bessely(v: ArrayLike, z: ArrayLike) -> np.ndarray:
        """
        Bessel function of the second kind of order v.
        
        Args:
            v: Order of the Bessel function
            z: Input values
        
        Returns:
            Bessel function of the second kind evaluated at z
        """
        return scipy_bessely(v, z)
    
    @staticmethod
    def besseli(v: ArrayLike, z: ArrayLike) -> np.ndarray:
        """
        Modified Bessel function of the first kind of order v.
        
        Args:
            v: Order of the modified Bessel function
            z: Input values
        
        Returns:
            Modified Bessel function of the first kind evaluated at z
        """
        return scipy_besseli(v, z)
    
    @staticmethod
    def besselk(v: ArrayLike, z: ArrayLike) -> np.ndarray:
        """
        Modified Bessel function of the second kind of order v.
        
        Args:
            v: Order of the modified Bessel function
            z: Input values
        
        Returns:
            Modified Bessel function of the second kind evaluated at z
        """
        return scipy_besselk(v, z)
    
    @staticmethod
    def airy(z: ArrayLike) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Airy functions and their derivatives.
        
        Airy functions satisfy the differential equation:
        y'' - z y = 0
        
        Args:
            z: Input values
        
        Returns:
            Tuple containing (Ai, Aip, Bi, Bip):
                Ai: Airy function of the first kind
                Aip: Derivative of Ai
                Bi: Airy function of the second kind
                Bip: Derivative of Bi
        """
        return scipy_airy(z)
    
    @staticmethod
    def hypergeom(a: ArrayLike, b: ArrayLike, c: ArrayLike, z: ArrayLike) -> np.ndarray:
        """
        Gaussian hypergeometric function 2F1(a, b; c; z).
        
        The hypergeometric function is defined as:
        2F1(a, b; c; z) = Σₙ=0^∞ (a)_n (b)_n / (c)_n * z^n / n!
        
        where (x)_n is the Pochhammer symbol.
        
        Args:
            a: First parameter
            b: Second parameter
            c: Third parameter
            z: Input values
        
        Returns:
            Hypergeometric function evaluated at z
        """
        return scipy_hyp2f1(a, b, c, z)
    
    @staticmethod
    def legendre(n: int, monic: bool = False) -> np.poly1d:
        """
        Legendre polynomial of degree n.
        
        Legendre polynomials satisfy the differential equation:
        (1 - z²) y'' - 2 z y' + n(n + 1) y = 0
        
        Args:
            n: Degree of the polynomial
            monic: If True, return monic polynomial
        
        Returns:
            Legendre polynomial as a numpy poly1d object
        """
        return scipy_legendre(n, monic=monic)
    
    @staticmethod
    def hermite(n: int, monic: bool = False) -> np.poly1d:
        """
        Hermite polynomial of degree n.
        
        Hermite polynomials satisfy the differential equation:
        y'' - 2 z y' + 2n y = 0
        
        Args:
            n: Degree of the polynomial
            monic: If True, return monic polynomial
        
        Returns:
            Hermite polynomial as a numpy poly1d object
        """
        return scipy_hermite(n, monic=monic)
    
    @staticmethod
    def laguerre(n: int, monic: bool = False) -> np.poly1d:
        """
        Laguerre polynomial of degree n.
        
        Laguerre polynomials satisfy the differential equation:
        z y'' + (1 - z) y' + n y = 0
        
        Args:
            n: Degree of the polynomial
            monic: If True, return monic polynomial
        
        Returns:
            Laguerre polynomial as a numpy poly1d object
        """
        return scipy_laguerre(n, monic=monic)
    
    @staticmethod
    def jacobi(n: int, alpha: float, beta: float, monic: bool = False) -> np.poly1d:
        """
        Jacobi polynomial of degree n.
        
        Args:
            n: Degree of the polynomial
            alpha: First parameter
            beta: Second parameter
            monic: If True, return monic polynomial
        
        Returns:
            Jacobi polynomial as a numpy poly1d object
        """
        return scipy_jacobi(n, alpha, beta, monic=monic)
    
    @staticmethod
    def zeta(x: ArrayLike, q: Optional[ArrayLike] = None) -> np.ndarray:
        """
        Riemann zeta function or Hurwitz zeta function.
        
        The Riemann zeta function is defined as:
        ζ(x) = Σₙ=1^∞ 1/n^x
        
        The Hurwitz zeta function is defined as:
        ζ(x, q) = Σₙ=0^∞ 1/(n + q)^x
        
        Args:
            x: Input values
            q: Hurwitz zeta function parameter (optional)
        
        Returns:
            Zeta function evaluated at x
        """
        return scipy_zeta(x, q)
    
    @staticmethod
    def polygamma(n: ArrayLike, x: ArrayLike) -> np.ndarray:
        """
        Polygamma function of order n.
        
        The polygamma function is defined as:
        ψ⁽ⁿ⁾(x) = dⁿ⁺¹/dxⁿ⁺¹ ln Γ(x)
        
        Args:
            n: Order of the polygamma function
            x: Input values
        
        Returns:
            Polygamma function evaluated at x
        """
        return scipy_polygamma(n, x)
    
    @staticmethod
    def digamma(x: ArrayLike) -> np.ndarray:
        """
        Digamma function (ψ⁰(x)).
        
        The digamma function is the logarithmic derivative of the gamma function:
        ψ(x) = d/dx ln Γ(x) = Γ'(x)/Γ(x)
        
        Args:
            x: Input values
        
        Returns:
            Digamma function evaluated at x
        """
        return scipy_digamma(x)
    
    @staticmethod
    def trigamma(x: ArrayLike) -> np.ndarray:
        """
        Trigamma function (ψ¹(x)).
        
        The trigamma function is the derivative of the digamma function:
        ψ¹(x) = d/dx ψ(x)
        
        Args:
            x: Input values
        
        Returns:
            Trigamma function evaluated at x
        """
        return scipy_polygamma(1, x)
    
    @staticmethod
    def gammainc(a: ArrayLike, x: ArrayLike) -> np.ndarray:
        """
        Regularized lower incomplete gamma function.
        
        Defined as:
        γ(a, x)/Γ(a) = (1/Γ(a)) ∫₀ˣ t^(a-1) e^(-t) dt
        
        Args:
            a: Shape parameter
            x: Input values
        
        Returns:
            Regularized lower incomplete gamma function
        """
        return gammainc(a, x)
    
    @staticmethod
    def gammaincc(a: ArrayLike, x: ArrayLike) -> np.ndarray:
        """
        Regularized upper incomplete gamma function.
        
        Defined as:
        Γ(a, x)/Γ(a) = (1/Γ(a)) ∫ₓ^∞ t^(a-1) e^(-t) dt
        
        Args:
            a: Shape parameter
            x: Input values
        
        Returns:
            Regularized upper incomplete gamma function
        """
        return gammaincc(a, x)
    
    @staticmethod
    def expi(x: ArrayLike) -> np.ndarray:
        """
        Exponential integral Ei(x).
        
        Defined as:
        Ei(x) = ∫_{-∞}^x e^t / t dt
        
        Args:
            x: Input values
        
        Returns:
            Exponential integral Ei(x)
        """
        return expi(x)
    
    @staticmethod
    def expn(n: ArrayLike, x: ArrayLike) -> np.ndarray:
        """
        Exponential integral E_n(x).
        
        Defined as:
        E_n(x) = ∫₁^∞ e^(-x t) / t^n dt
        
        Args:
            n: Order of the exponential integral
            x: Input values
        
        Returns:
            Exponential integral E_n(x)
        """
        return expn(n, x)
    
    @staticmethod
    def spherical_harmonic(m: int, n: int, theta: ArrayLike, phi: ArrayLike) -> np.ndarray:
        """
        Spherical harmonic function Y_n^m(theta, phi).
        
        Args:
            m: Order (|m| <= n)
            n: Degree (non-negative integer)
            theta: Polar angle (0 <= theta <= pi)
            phi: Azimuthal angle (0 <= phi <= 2pi)
        
        Returns:
            Spherical harmonic values
        """
        from scipy.special import sph_harm
        return sph_harm(m, n, phi, theta)

# Create an instance for easy access
sf = SpecialFunction()

# Core interface functions (matching scipy.special interface)
gamma = sf.gamma
factorial = sf.factorial
beta = sf.beta
erf = sf.erf
erfc = sf.erfc
besselj = sf.besselj
bessely = sf.bessely
besseli = sf.besseli
besselk = sf.besselk
airy = sf.airy
hypergeom = sf.hypergeom
legendre = sf.legendre
hermite = sf.hermite
laguerre = sf.laguerre
jacobi = sf.jacobi
zeta = sf.zeta
polygamma = sf.polygamma
digamma = sf.digamma
trigamma = sf.trigamma

# Additional commonly used special functions
j0 = j0
j1 = j1
jn = jn
y0 = y0
y1 = y1
yn = yn
i0 = i0
i1 = i1
k0 = k0
k1 = k1
kn = kn
loggamma = loggamma
gammainc = gammainc
gammaincc = gammaincc
erfinv = erfinv
erfcinv = erfcinv
expi = expi
expn = expn
spherical_harmonic = sf.spherical_harmonic
