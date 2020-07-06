import numpy as np
import cython
from cython.parallel import parallel, prange
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy, memset

cimport scipy.linalg.cython_lapack as cython_lapack
cimport scipy.linalg.cython_blas as cython_blas


cdef inline void axpy(int *n, float *da, float *dx, int *incx, float *dy, 
                      int *incy) nogil:
    cython_blas.saxpy(n, da, dx, incx, dy, incy)


cdef inline void posv(char *u, int *n, int *nrhs, float *a, int *lda, 
                      float *b, int *ldb, int *info) nogil:
    cython_lapack.sposv(u, n, nrhs, a, lda, b, ldb, info)


cdef inline void symv(char *u, int *n, float *alpha, float *a, int *lda, 
                      float *x, int *incx, float *beta, float *y, 
                      int *incy) nogil:
    cython_blas.ssymv(u, n, alpha, a, lda, x, incx, beta, y, incy)


cdef inline float dot(int *n, float *sx, int *incx, float *sy, 
                      int *incy) nogil:
    return cython_blas.sdot(n, sx, incx, sy, incy)


cdef inline void scal(int *n, float *sa, float *sx, int *incx) nogil:
    cython_blas.sscal(n, sa, sx, incx)


def als_update(interaction, X, Y, reg, task, use_cg=True, 
               num_threads=1, cg_steps=3):
    if task == "rating" and use_cg:
        _least_squares_cg(interaction.indices, interaction.indptr, 
            interaction.data, X, Y, reg, num_threads, 0, cg_steps)
    elif task == "rating" and not use_cg:
        _least_squares(interaction.indices, interaction.indptr, 
            interaction.data, X, Y, reg, num_threads, 0)
    elif task == "ranking" and use_cg:
        _least_squares_cg(interaction.indices, interaction.indptr, 
            interaction.data, X, Y, reg, num_threads, 1, cg_steps)
    elif task == "ranking" and not use_cg:
        _least_squares(interaction.indices, interaction.indptr, 
            interaction.data, X, Y, reg, num_threads, 1)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void _least_squares(const int[:] indices, const int[:] indptr, 
    const float[:] data, float[:, ::1] X, float[:, ::1] Y, double reg, 
    int num_threads, int implicit):
    cdef int n_x = X.shape[0], embed_size = X.shape[1]
    cdef int m, i, j, index, err, one = 1
    cdef float rating, confidence, temp

    cdef float[:, ::1] initialA
    if implicit > 0:
        initialA = np.dot(np.transpose(Y), Y) + reg * np.eye(embed_size, dtype=np.single)
    else:
        initialA = reg * np.eye(embed_size, dtype=np.single)
    cdef float[:] initialB = np.zeros(embed_size, dtype=np.single)
    cdef float *A
    cdef float *b

    with nogil, parallel(num_threads=num_threads):
        A = <float *> malloc(sizeof(float) * embed_size * embed_size)
        b = <float *> malloc(sizeof(float) * embed_size)
        try:
            for m in prange(n_x, schedule="guided"):
                memcpy(A, &initialA[0, 0], sizeof(float) * embed_size * embed_size)
                memcpy(b, &initialB[0], sizeof(float) * embed_size)

                for index in range(indptr[m], indptr[m+1]):
                    if implicit > 0:
                        i = indices[index]
                        confidence = data[index]
                        # compute partial A = Yu^T @ Cu @ Yu + lambda * I
                        for j in range(embed_size):
                            temp = (confidence - 1) * Y[i, j]
                            axpy(&embed_size, &temp, &Y[i, 0], &one, 
                                 A + j * embed_size, &one)

                        # compute partial b = Yu^T @ Ru
                        axpy(&embed_size, &confidence, &Y[i, 0], &one, b, &one)
                    else:
                        i = indices[index]
                        rating = data[index]
                        # compute partial A = Yu^T @ Yu + lambda * I
                        for j in range(embed_size):
                            axpy(&embed_size, &Y[i, j], &Y[i, 0], &one, A + j * embed_size, &one)

                        # compute partial b = Yu^T @ Ru
                        axpy(&embed_size, &rating, &Y[i, 0], &one, b, &one)

                err = 0
                # solve Ax = b, x = A^-1 * b
                posv("U", &embed_size, &one, A, &embed_size, b, &embed_size, &err)
                if not err:
                    memcpy(&X[m, 0], b, sizeof(float) * embed_size)
                else:
                    with gil:
                        raise ValueError(f"cython_lapack.posv failed (err={err}) on row {m}. "
                                          "Try increasing the regularization parameter.")

        finally:
            free(A)
            free(b)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void _least_squares_cg(const int[:] indices, const int[:] indptr, 
    const float[:] data, float[:, ::1] X, float[:, ::1] Y, double reg, 
    int num_threads, int implicit, int cg_steps):
    cdef int n_x = X.shape[0], embed_size = X.shape[1]
    cdef int m, i, j, index, err, one = 1
    cdef float rating, confidence, temp, rsold, rsnew, ak
    cdef float zero = 0.0

    cdef float[:, ::1] initialA
    if implicit > 0:
        initialA = YtY = np.dot(np.transpose(Y), Y) + reg * np.eye(embed_size, dtype=np.single)
    else:
        initialA = reg * np.eye(embed_size, dtype=np.single)

    cdef float *x
    cdef float *p
    cdef float *r
    cdef float *Ap

    with nogil, parallel(num_threads=num_threads):
        Ap = <float *> malloc(sizeof(float) * embed_size)
        p = <float *> malloc(sizeof(float) * embed_size)
        r = <float *> malloc(sizeof(float) * embed_size)
        try:
            for m in prange(n_x, schedule="guided"):
                x = &X[m, 0]

                temp = -1.0
                # compute residual r = b - Ax
                # first step: r = -(YtY + lambdaI)^T @ x  or  -(lambdaI @ x)
                symv("U", &embed_size, &temp, &initialA[0, 0], &embed_size, x, &one, &zero, r, &one)
                for index in range(indptr[m], indptr[m+1]):
                    if implicit > 0:
                        i = indices[index]
                        confidence = data[index]
                        # second step: r += (c - (c-1)y @ x) * y
                        temp = confidence - (confidence - 1) * dot(&embed_size, &Y[i, 0], &one, x, &one)
                        axpy(&embed_size, &temp, &Y[i, 0], &one, r, &one)
                    else:
                        i = indices[index]
                        rating = data[index]
                        # second step: r += (rating - y @ x) * y
                        temp = rating - dot(&embed_size, &Y[i, 0], &one, x, &one)
                        axpy(&embed_size, &temp, &Y[i, 0], &one, r, &one)

                memcpy(p, r, sizeof(float) * embed_size)
                rsold = dot(&embed_size, r, &one, r, &one)
                if rsold < 1e-10:
                    continue

                for j in range(cg_steps):
                    temp = 1.0
                    # compute Ap
                    # first step: Ap = -(YtY + lambdaI)^T @ p  or  -(lambdaI @ p)
                    symv("U", &embed_size, &temp, &initialA[0, 0], &embed_size, p, &one, &zero, Ap, &one)
                    for index in range(indptr[m], indptr[m+1]):
                        if implicit > 0:
                            i = indices[index]
                            confidence = data[index]
                            # second step: Ap += (c-1) * (y @ x) * y
                            temp = (confidence - 1) * dot(&embed_size, &Y[i, 0], &one, p, &one)
                            axpy(&embed_size, &temp, &Y[i, 0], &one, Ap, &one)
                        else:
                            i = indices[index]
                            # second step: Ap += (y @ x) * y
                            temp = dot(&embed_size, &Y[i, 0], &one, p, &one)
                            axpy(&embed_size, &temp, &Y[i, 0], &one, Ap, &one)

                    # ak = rsold / p.dot(Ap)
                    ak = rsold / dot(&embed_size, p, &one, Ap, &one)
                    # x += ak * p
                    axpy(&embed_size, &ak, p, &one, x, &one)
                    # r -= alpha * Ap
                    temp = ak * -1
                    axpy(&embed_size, &temp, Ap, &one, r, &one)

                    rsnew = dot(&embed_size, r, &one, r, &one)
                    if rsnew < 1e-10:
                        break

                    # p = r + (rsnew/rsold) * p
                    temp = rsnew / rsold
                    scal(&embed_size, &temp, p, &one)
                    temp = 1.0
                    axpy(&embed_size, &temp, r, &one, p, &one)
                    rsold = rsnew

        finally:
            free(Ap)
            free(p)
            free(r)

