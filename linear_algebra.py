"""
Linear Algebra Module
=====================

This module provides classes and functions for linear algebra computations,
including solving linear systems, matrix decompositions, eigenvalue problems,
and least squares fitting.

Classes:
    LinearSystemSolver: Solve linear systems Ax = b using various methods
    MatrixDecomposer: Perform matrix decompositions (LU, QR, SVD, etc.)

Functions:
    solve: Solve linear system Ax = b
    least_squares: Solve least squares problem min ||Ax - b||
    eig: Compute eigenvalues and eigenvectors of a square matrix
    svd: Compute singular value decomposition of a matrix
"""

import numpy as np
from typing import Tuple, Union, Optional

ArrayLike = Union[np.ndarray, list, tuple]

class LinearSystemSolver:
    """
    A class for solving linear systems Ax = b using various methods.
    
    Methods:
        lu: Solve using LU decomposition
        qr: Solve using QR decomposition
        cholesky: Solve using Cholesky decomposition (for positive definite matrices)
        gauss_seidel: Solve using Gauss-Seidel iterative method
        jacobi: Solve using Jacobi iterative method
    """
    
    def __init__(self, A: ArrayLike):
        """
        Initialize the LinearSystemSolver with coefficient matrix A.
        
        Args:
            A: Coefficient matrix of shape (n, n)
        """
        self.A = np.asarray(A, dtype=np.float64)
        self.n = self.A.shape[0]
        
    def lu(self, b: ArrayLike) -> np.ndarray:
        """Solve linear system using LU decomposition."""
        b = np.asarray(b, dtype=np.float64)
        return np.linalg.solve(self.A, b)
    
    def qr(self, b: ArrayLike) -> np.ndarray:
        """Solve linear system using QR decomposition."""
        b = np.asarray(b, dtype=np.float64)
        Q, R = np.linalg.qr(self.A)
        return np.linalg.solve(R, Q.T @ b)
    
    def cholesky(self, b: ArrayLike) -> np.ndarray:
        """Solve linear system using Cholesky decomposition."""
        b = np.asarray(b, dtype=np.float64)
        L = np.linalg.cholesky(self.A)
        y = np.linalg.solve(L, b)
        return np.linalg.solve(L.T, y)
    
    def gauss_seidel(self, b: ArrayLike, tol: float = 1e-10, max_iter: int = 1000) -> np.ndarray:
        """Solve linear system using Gauss-Seidel iterative method."""
        b = np.asarray(b, dtype=np.float64)
        x = np.zeros_like(b)
        
        for _ in range(max_iter):
            x_new = x.copy()
            for i in range(self.n):
                s1 = np.dot(self.A[i, :i], x_new[:i])
                s2 = np.dot(self.A[i, i+1:], x[i+1:])
                x_new[i] = (b[i] - s1 - s2) / self.A[i, i]
            
            if np.linalg.norm(x_new - x) < tol:
                return x_new
            x = x_new
        
        raise ValueError("Gauss-Seidel method did not converge")
    
    def jacobi(self, b: ArrayLike, tol: float = 1e-10, max_iter: int = 1000) -> np.ndarray:
        """Solve linear system using Jacobi iterative method."""
        b = np.asarray(b, dtype=np.float64)
        x = np.zeros_like(b)
        D = np.diag(self.A)
        R = self.A - np.diag(D)
        
        for _ in range(max_iter):
            x_new = (b - R @ x) / D
            if np.linalg.norm(x_new - x) < tol:
                return x_new
            x = x_new
        
        raise ValueError("Jacobi method did not converge")

class MatrixDecomposer:
    """
    A class for performing matrix decompositions.
    
    Methods:
        lu: LU decomposition
        qr: QR decomposition
        cholesky: Cholesky decomposition
        eig: Eigenvalue decomposition
        svd: Singular value decomposition
    """
    
    def __init__(self, A: ArrayLike):
        """
        Initialize the MatrixDecomposer with matrix A.
        
        Args:
            A: Input matrix of shape (m, n)
        """
        self.A = np.asarray(A, dtype=np.float64)
    
    def lu(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """LU decomposition: A = P L U."""
        P, L, U = np.linalg.lu(self.A)
        return P, L, U
    
    def qr(self) -> Tuple[np.ndarray, np.ndarray]:
        """QR decomposition: A = Q R."""
        Q, R = np.linalg.qr(self.A)
        return Q, R
    
    def cholesky(self) -> np.ndarray:
        """Cholesky decomposition: A = L L^T (for positive definite matrices)."""
        return np.linalg.cholesky(self.A)
    
    def eig(self) -> Tuple[np.ndarray, np.ndarray]:
        """Eigenvalue decomposition: A = V Λ V^(-1)."""
        eigenvalues, eigenvectors = np.linalg.eig(self.A)
        return eigenvalues, eigenvectors
    
    def svd(self, full_matrices: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Singular value decomposition: A = U Σ V^T."""
        U, S, Vt = np.linalg.svd(self.A, full_matrices=full_matrices)
        return U, S, Vt.T

# Core interface functions
def solve(A: ArrayLike, b: ArrayLike, method: str = 'lu') -> np.ndarray:
    """
    Solve linear system Ax = b.
    
    Args:
        A: Coefficient matrix of shape (n, n)
        b: Right-hand side vector of shape (n,)
        method: Solver method ('lu', 'qr', 'cholesky', 'gauss_seidel', 'jacobi')
    
    Returns:
        Solution vector x of shape (n,)
    """
    solver = LinearSystemSolver(A)
    
    if method == 'lu':
        return solver.lu(b)
    elif method == 'qr':
        return solver.qr(b)
    elif method == 'cholesky':
        return solver.cholesky(b)
    elif method == 'gauss_seidel':
        return solver.gauss_seidel(b)
    elif method == 'jacobi':
        return solver.jacobi(b)
    else:
        raise ValueError(f"Unknown method: {method}")

def least_squares(A: ArrayLike, b: ArrayLike) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """
    Solve least squares problem min ||Ax - b||.
    
    Args:
        A: Design matrix of shape (m, n)
        b: Observation vector of shape (m,)
    
    Returns:
        x: Solution vector of shape (n,)
        residuals: Sum of squared residuals
        rank: Rank of matrix A
        singular_values: Singular values of matrix A
    """
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    return x, residuals, rank, s

def eig(A: ArrayLike, right: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Compute eigenvalues and eigenvectors of a square matrix.
    
    Args:
        A: Square matrix of shape (n, n)
        right: If True, compute right eigenvectors (default)
    
    Returns:
        eigenvalues: Array of eigenvalues
        eigenvectors: Array of eigenvectors (if right=True)
    """
    A = np.asarray(A, dtype=np.float64)
    if right:
        eigenvalues, eigenvectors = np.linalg.eig(A)
        return eigenvalues, eigenvectors
    else:
        eigenvalues = np.linalg.eigvals(A)
        return eigenvalues, None

def svd(A: ArrayLike, full_matrices: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute singular value decomposition of a matrix.
    
    Args:
        A: Input matrix of shape (m, n)
        full_matrices: If True, return full U and V matrices
    
    Returns:
        U: Left singular vectors of shape (m, m) or (m, k)
        S: Singular values of shape (k,)
        V: Right singular vectors of shape (n, n) or (n, k)
    """
    decomposer = MatrixDecomposer(A)
    return decomposer.svd(full_matrices=full_matrices)

def matrix_inverse(A: ArrayLike) -> np.ndarray:
    """
    Compute the inverse of a square matrix.
    
    Args:
        A: Square matrix of shape (n, n)
    
    Returns:
        Inverse matrix of shape (n, n)
    """
    A = np.asarray(A, dtype=np.float64)
    return np.linalg.inv(A)

def determinant(A: ArrayLike) -> float:
    """
    Compute the determinant of a square matrix.
    
    Args:
        A: Square matrix of shape (n, n)
    
    Returns:
        Determinant value
    """
    A = np.asarray(A, dtype=np.float64)
    return np.linalg.det(A)
