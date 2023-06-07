#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;

void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    int batch_n = int(m+1)/batch;
    for(int i=0; i<batch_n; i++) {
      
      // submatrix of examples; shape= [si:ei] * n
      int si = i * batch;
      int ei = std::min(si+batch, m);

      // shape of batch matrix data
      int bM = ei-si;
      int bN = n;
      int bK = k;
      
      // STEP1: b_xt = np.exp(b_X @ theta)
      float b_xt[bM*bK]; // store 2d for BX@Theta [bm,k]
      for(int bm=0; bm<bM ; bm++) { // matrix mult and exp(xij)
        for(int bk=0; bk<bK; bk++) {
          float dot_v = 0;
          for(int bn=0; bn<bN; bn++) {
            dot_v += X[(si+bm)*bN + bn] * theta[bn*bK + bk];
          }
          // exp for item of matrix(b_X @ theta)
          b_xt[bm*bK + bk]= exp(dot_v);
        }
      }
    
      // STEP2: b_xt_norm = np.sum(b_xt, axis=1, keepdims=True)
      // b_Z = b_xt / b_xt_norm
      float b_Z[bM*bK];
      for(int bm=0; bm<bM ; bm++) {
        float row_norm = 0;
        // iter one b_xt row 
        for(int bk=0; bk<bK; bk++) {
          row_norm += b_xt[bm*bK+ bk];
        }
        // iter one b_Z row, set normed v
        for(int bk=0; bk<bK; bk++) {
          b_Z[bm*bK+bk] = b_xt[bm*bK+bk]/row_norm;
        }
      }
      
      // STEP3: label to one_hot
      // b_Iy = np.zeros(b_Z.shape)
      // b_Iy[np.arange(b_Z.shape[0]), b_y] = 1
      float b_Iy[bM*bK];
      for(int bm=0; bm<bM; bm++) {
        for(int bk=0; bk<bK; bk++) {
          if (bk == y[si+bm]) {
            b_Iy[bm*bK+ bk] = 1;
          } else {
            b_Iy[bm*bK+ bk] = 0;
          }
        }
      }

      // STEP4: Delta = b_X.T @ (b_Z- b_Iy) 
      float Delta[bN*bK];
      for(int bn=0; bn<bN; bn++) {
        for(int bk=0; bk<bK; bk++) {
          float dot_v = 0;
          for(int bm=0; bm< bM; bm++) {
            dot_v += X[(si+bm)*bN+bn] * (b_Z[bm*bK+bk] - b_Iy[bm*bK+bk]);
          }
          Delta[bn*bK + bk] = dot_v;
        }
      }

      // STEP5: update grad of theta
      // theta -= lr * Delta // b_X.shape[0]
      for(int bn=0; bn<bN; bn++) {
        for(int bk=0; bk<bK; bk++) {
        theta[bn*bK + bk] -= lr/bM * Delta[bn*bK + bk];
        }
      }
    }

    return;
    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
