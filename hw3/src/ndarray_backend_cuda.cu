#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <sstream>

namespace needle {
namespace cuda {

#define BASE_THREAD_NUM 256

typedef float scalar_t;
const size_t ELEM_SIZE = sizeof(scalar_t);
typedef ssize_t ptrdiff_t;

struct CudaArray {
  CudaArray(const size_t size) {
    cudaError_t err = cudaMalloc(&ptr, size * ELEM_SIZE);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    this->size = size;
  }
  ~CudaArray() { cudaFree(ptr); }
  size_t ptr_as_int() { return (size_t)ptr; }
  
  scalar_t* ptr;
  size_t size;
};

struct CudaDims {
  dim3 block, grid;
};

CudaDims CudaOneDim(size_t size) {
  /**
   * Utility function to get cuda dimensions for 1D call
   */
  CudaDims dim;
  size_t num_blocks = (size + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM;
  dim.block = dim3(BASE_THREAD_NUM, 1, 1);
  dim.grid = dim3(num_blocks, 1, 1);
  return dim;
}

#define MAX_VEC_SIZE 8
struct CudaVec {
  uint32_t size;
  int32_t data[MAX_VEC_SIZE];
};

CudaVec VecToCuda(const std::vector<int32_t>& x) {
  CudaVec shape;
  if (x.size() > MAX_VEC_SIZE) throw std::runtime_error("Exceeded CUDA supported max dimesions");
  shape.size = x.size();
  for (size_t i = 0; i < x.size(); i++) {
    shape.data[i] = x[i];
  }
  return shape;
}

////////////////////////////////////////////////////////////////////////////////
// Fill call
////////////////////////////////////////////////////////////////////////////////

__global__ void FillKernel(scalar_t* out, scalar_t val, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = val;
}

void Fill(CudaArray* out, scalar_t val) {
  CudaDims dim = CudaOneDim(out->size);
  FillKernel<<<dim.grid, dim.block>>>(out->ptr, val, out->size);
}

////////////////////////////////////////////////////////////////////////////////
// Compact and setitem cals
////////////////////////////////////////////////////////////////////////////////

// Untility function to convert contiguous index i to memory location from strides

/**
* 在 CUDA 编程中，__global__ 和 __device__ 是两种函数类型修饰符。
* - __global__：这个修饰符表示该函数是一个核函数（kernel function），可以从 CPU 上调用并在 GPU 上执行。这种函数不能返回值，必须返回 void。
* - __device__：这个修饰符表示该函数是一个设备函数（device function），只能从 GPU 上调用并在 GPU 上执行。这种函数可以返回值。
* 这两种函数类型修饰符是 CUDA 编程模型的一部分，用于区分函数的执行位置和调用方式。 
*/


__device__ size_t calc_position(size_t gid, const CudaVec shape, const CudaVec strides) {
  /**
   * gid是out的位置，需要通过shape and strides 还原出a的位置
   *  step1: gid 到matrix（i，j）的映射
   *  step2: 通过stride计算new（i，j）对应的元素在old matrix的内存位置
  */
  size_t N = shape.size;

  CudaVec new_strides = CudaVec();
  new_strides.size = N;

  // calc new_strides
  new_strides.data[N-1] = 1;
  for(int i=N-2; i>=0; i--) {
    new_strides.data[i] = new_strides.data[i+1] * shape.data[i+1];
  }

  // calc index and old position
  size_t position = 0;
  size_t left = gid;

  for(int i=0; i<N; i++) {
    size_t index_i = left / new_strides.data[i];    
    position += index_i * strides.data[i]; // update positon

    left = left % new_strides.data[i];
    if(left==0) return position; // 提前退出
  }
  return position; // 应该走不到这，new_strides[-1]=1
}


__global__ void CompactKernel(const scalar_t* a, scalar_t* out, size_t size, CudaVec shape,
                              CudaVec strides, size_t offset) {
  /**
   * The CUDA kernel for the compact opeation.  This should effectively map a single entry in the 
   * non-compact input a, to the corresponding item (at location gid) in the compact array out.
   * 
   * Args:
   *   a: non-compact represntation of the array, given as input
   *   out: compact version of the array to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *a* array (not out, which has compact strides)
   *   offset: offset of the *a* array (not out, which has zero offset, being compact)
   */
  ssize_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  /// BEGIN YOUR SOLUTION
  if (gid < size) {
      size_t position = offset + calc_position(gid, shape, strides);
      out[gid] = a[position];
  }
  /// END YOUR SOLUTION
}

void Compact(const CudaArray& a, CudaArray* out, std::vector<int32_t> shape,
             std::vector<int32_t> strides, size_t offset) {
  /**
   * Compact an array in memory.  Unlike the C++ version, in CUDA this will primarily call the 
   * relevant CUDA kernel.  In this case, we illustrate how you should set this up (i.e., we give 
   * you the code for this fuction, and also the prototype for the CompactKernel() function).  For
   * the functions after this, however, you'll need to define these kernels as you see fit to 
   * execute the underlying function.
   * 
   * Args:
   *   a: non-compact represntation of the array, given as input
   *   out: compact version of the array to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *a* array (not out, which has compact strides)
   *   offset: offset of the *a* array (not out, which has zero offset, being compact)
   */

  // Nothing needs to be added here
  CudaDims dim = CudaOneDim(out->size);
  CompactKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, VecToCuda(shape),
                                         VecToCuda(strides), offset);
}


__global__ void EwiseSetitemKernel(const scalar_t* a, scalar_t* out, size_t size, CudaVec shape,
                              CudaVec strides, size_t offset) {
  /**
   * 
   * Args:
   *   a: CUDA pointer to a array
   *   out: CUDA point to out array
   *   size: size of out array
   *   shape: vector of shapes of a and out arrays (of type CudaVec, for past passing to CUDA kernel)
   *   strides: vector of strides of out array
   *   offset: offset of out array
   */
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  /// BEGIN YOUR SOLUTION
  // calac index by strides
  if (gid < size) {
      size_t position = offset + calc_position(gid, shape, strides);
      out[position] = a[gid];
  }

  /// END YOUR SOLUTION
}


void EwiseSetitem(const CudaArray& a, CudaArray* out, std::vector<int32_t> shape,
                  std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items in a (non-compact) array using CUDA.  Yyou will most likely want to implement a
   * EwiseSetitemKernel() function, similar to those above, that will do the actual work.
   * 
   * Args:
   *   a: _compact_ array whose items will be written to out
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *out* array (not a, which has compact strides)
   *   offset: offset of the *out* array (not a, which has zero offset, being compact)
   */
  /// BEGIN YOUR SOLUTION
  CudaDims dim = CudaOneDim(out->size);
  EwiseSetitemKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, a.size, VecToCuda(shape),
                                         VecToCuda(strides), offset); // todo 注意：a.size，使用out.size有问题
  /// END YOUR SOLUTION
}


__global__ void ScalarSetitemKernel(scalar_t val, scalar_t* out, size_t size, CudaVec shape,
                              CudaVec strides, size_t offset) {
  /**
   *
   * Args:
   *   a: CUDA pointer to a array
   *   out: CUDA point to out array
   *   size: size of out array
   *   shape: vector of shapes of a and out arrays (of type CudaVec, for past passing to CUDA kernel)
   *   strides: vector of strides of out array
   *   offset: offset of out array
   */
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  /// BEGIN YOUR SOLUTION
  // calac index by strides
  size_t position = offset + calc_position(gid, shape, strides);
  if (gid < size) out[position] = val;
  /// END YOUR SOLUTION
}


void ScalarSetitem(size_t size, scalar_t val, CudaArray* out, std::vector<int32_t> shape,
                   std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items is a (non-compact) array
   * 
   * Args:
   *   size: number of elements to write in out array (note that this will note be the same as
   *         out.size, because out is a non-compact subset array);  it _will_ be the same as the 
   *         product of items in shape, but covenient to just pass it here.
   *   val: scalar value to write to
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension of out
   *   strides: strides of the out array
   *   offset: offset of the out array
   */
  /// BEGIN YOUR SOLUTION
  CudaDims dim = CudaOneDim(out->size);
  ScalarSetitemKernel<<<dim.grid, dim.block>>>(val, out->ptr, size, VecToCuda(shape),
                                         VecToCuda(strides), offset);
  /// END YOUR SOLUTION
}

////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////

__global__ void EwiseAddKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] + b[gid];
}

void EwiseAdd(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  /**
   * Add together two CUDA array
   */
  CudaDims dim = CudaOneDim(out->size);
  EwiseAddKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarAddKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] + val;
}

void ScalarAdd(const CudaArray& a, scalar_t val, CudaArray* out) {
  /**
   * Add together a CUDA array and a scalar value.
   */
  CudaDims dim = CudaOneDim(out->size);
  ScalarAddKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

/**
 * In the code the follows, use the above template to create analogous elementise
 * and and scalar operators for the following functions.  See the numpy backend for
 * examples of how they should work.
 *   - EwiseMul, ScalarMul
 *   - EwiseDiv, ScalarDiv
 *   - ScalarPower
 *   - EwiseMaximum, ScalarMaximum
 *   - EwiseEq, ScalarEq
 *   - EwiseGe, ScalarGe
 *   - EwiseLog
 *   - EwiseExp
 *   - EwiseTanh
 *
 * If you implement all these naively, there will be a lot of repeated code, so
 * you are welcome (but not required), to use macros or templates to define these
 * functions (however you want to do so, as long as the functions match the proper)
 * signatures above.
 */

/// BEGIN YOUR SOLUTION
__global__ void EwiseMulKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] * b[gid];
}

void EwiseMul(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  /**
   * Add together two CUDA array
   */
  CudaDims dim = CudaOneDim(out->size);
  EwiseMulKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarMulKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] * val;
}

void ScalarMul(const CudaArray& a, scalar_t val, CudaArray* out) {
  /**
   * Add together a CUDA array and a scalar value.
   */
  CudaDims dim = CudaOneDim(out->size);
  ScalarMulKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}


__global__ void EwiseDivKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] / b[gid];
}

void EwiseDiv(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  /**
   * Add together two CUDA array
   */
  CudaDims dim = CudaOneDim(out->size);
  EwiseDivKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarDivKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] / val;
}

void ScalarDiv(const CudaArray& a, scalar_t val, CudaArray* out) {
  /**
   * Add together a CUDA array and a scalar value.
   */
  CudaDims dim = CudaOneDim(out->size);
  ScalarDivKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

__global__ void ScalarPowerKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = std::pow(a[gid], val);
}


void ScalarPower(const CudaArray& a, scalar_t val, CudaArray* out) {
  /**
   * Add together a CUDA array and a scalar value.
   */
  CudaDims dim = CudaOneDim(out->size);
  ScalarPowerKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

__global__ void EwiseMaximumKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid]>b[gid] ? a[gid] : b[gid];
}

void EwiseMaximum(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  /**
   * Add together two CUDA array
   */
  CudaDims dim = CudaOneDim(out->size);
  EwiseMaximumKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarMaximumKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid]>val ? a[gid] : val;
}

void ScalarMaximum(const CudaArray& a, scalar_t val, CudaArray* out) {
  /**
   * Add together a CUDA array and a scalar value.
   */
  CudaDims dim = CudaOneDim(out->size);
  ScalarMaximumKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

__global__ void EwiseEqKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] == b[gid];
}

void EwiseEq(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  /**
   * Add together two CUDA array
   */
  CudaDims dim = CudaOneDim(out->size);
  EwiseEqKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarEqKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] == val;
}

void ScalarEq(const CudaArray& a, scalar_t val, CudaArray* out) {
  /**
   * Add together a CUDA array and a scalar value.
   */
  CudaDims dim = CudaOneDim(out->size);
  ScalarEqKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

__global__ void EwiseGeKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] >= b[gid];
}

void EwiseGe(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  /**
   * Add together two CUDA array
   */
  CudaDims dim = CudaOneDim(out->size);
  EwiseGeKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarGeKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] >= val;
}

void ScalarGe(const CudaArray& a, scalar_t val, CudaArray* out) {
  /**
   * Add together a CUDA array and a scalar value.
   */
  CudaDims dim = CudaOneDim(out->size);
  ScalarGeKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

__global__ void EwiseLogKernel(const scalar_t* a, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = std::log(a[gid]);
}

void EwiseLog(const CudaArray& a, CudaArray* out) {
  /**
   * Add together two CUDA array
   */
  CudaDims dim = CudaOneDim(out->size);
  EwiseLogKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size);
}

__global__ void EwiseExpKernel(const scalar_t* a, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = std::exp(a[gid]);
}

void EwiseExp(const CudaArray& a, CudaArray* out) {
  /**
   * Add together two CUDA array
   */
  CudaDims dim = CudaOneDim(out->size);
  EwiseExpKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size);
}

__global__ void EwiseTanhKernel(const scalar_t* a, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = std::tanh(a[gid]);
}

void EwiseTanh(const CudaArray& a, CudaArray* out) {
  /**
   * Add together two CUDA array
   */
  CudaDims dim = CudaOneDim(out->size);
  EwiseTanhKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size);
}

/// END YOUR SOLUTION

////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////

// # 避免corner case，保证L可以被V整除
#define V 4
#define L 32
#define S 4


__global__ void MatmulKernel(const scalar_t* A, const scalar_t* B, scalar_t* Out, uint32_t M, uint32_t N, uint32_t P) {
    __shared__ scalar_t sA[L][S];
    __shared__ scalar_t sB[S][L];

    scalar_t c[V][V] = {0};
    scalar_t a[V],b[V];
    
    size_t xblock = blockIdx.x;
    size_t yblock = blockIdx.y;

    size_t nthreads =  blockDim.y * blockDim.x; // block内线程总数
    size_t tid = threadIdx.x * blockDim.y + threadIdx.y;  // block内二维线程的 local_id 

    size_t xbase = blockIdx.x * blockDim.x + threadIdx.x; // thread global x_id
    size_t ybase = blockIdx.y * blockDim.y + threadIdx.y; // thread global y_id

    for (size_t kn=0; kn<N; kn+=S) { // N轴维度,c[V][V]全局累加
      // 同一个block内的共享变量赋值
      for (size_t j=0; j<L*S/nthreads; j++) {
        // sA[:, :] = A[k : k + S, yblock * L : yblock * L + L];
        size_t li  = (j*nthreads + tid)/S;
        size_t sj = (j*nthreads + tid)%S;
        if ((xblock*L+li) < M && (kn+sj) < N) { // 处理边界情况
           sA[li][sj] = A[(xblock*L+li)*N + (kn+sj)]; // A[xblock*L+li][k0+sj]
        }

        // sB[:, :] = B[k : k + S, xblock * L : xblock * L + L];
        size_t si = (j*nthreads + tid)/L;
        size_t lj = (j*nthreads + tid)%L;
        if ((kn+si)<N && (yblock*L+lj)<P) {
          sB[si][lj] = B[(kn+si)*P+(yblock*L+lj)]; // B[kn+si][yblock*L+lj]
        }
      }
      __syncthreads();

      // 各线程同步结束后，bolck内使用sA、sB计算各自线程相关的c[V][V]. 注意:c是重复使用的,可能存在废值。
      for (size_t ks=0;ks<S;ks++) { // S轴维度,c[V][V]局部累加
        // local mem到每个线程寄存器的TILE[V][V]的优化        
        if ((kn+ks)>=N) {continue;} // 注意: ks超边界，跳过

        // 寄存器a[:],b[:]赋值
        for (size_t i=0;i<V;i++) {
          // a[:] = sA[threadIdx.y * V : threadIdx.y * V + V, ks];
          a[i] =  (xbase*V+i) < M ? sA[threadIdx.x*V+i][ks] : 0;
          // b[:] = sB[ks, threadIdx.x * V : threadIdx.x * V + V];
          b[i] = (ybase*V+i) < P ? sB[ks][threadIdx.y*V+i] : 0;
        }
        // 局部累加c[:][:]
        for (size_t x=0;x<V;x++) {
          for (size_t y=0;y<V;y++) {
            c[x][y] += a[x]*b[y];
          }
        }
      }
    }

    // Out[ybase * V : ybase*V + V, xbase*V : xbase*V + V] = c[:];
    for (size_t i=0; i<V; i++) {
      size_t oi = xbase*V+i;
      if (oi < M) { // check legal oi
        for (size_t j=0; j<V; j++) {
          size_t oj = ybase*V+j;
          if (oj<P) { // check legal oj
            Out[oi*P+oj] = c[i][j];   // Out[xbase*V+i][ybase*V+j] = c[i][j]
          } 
        }
      }
    }

}

void Matmul(const CudaArray& a, const CudaArray& b, CudaArray* out, uint32_t M, uint32_t N,
            uint32_t P) {
  /**
   * Multiply two (compact) matrices into an output (also comapct) matrix.  You will want to look
   * at the lecture and notes on GPU-based linear algebra to see how to do this.  Since ultimately
   * mugrade is just evaluating correctness, you _can_ implement a version that simply parallelizes
   * over (i,j) entries in the output array.  However, to really get the full benefit of this
   * problem, we would encourage you to use cooperative fetching, shared memory register tiling, 
   * and other ideas covered in the class notes.  Note that unlike the tiled matmul function in
   * the CPU backend, here you should implement a single function that works across all size
   * matrices, whether or not they are a multiple of a tile size.  As with previous CUDA
   * implementations, this function here will largely just set up the kernel call, and you should
   * implement the logic in a separate MatmulKernel() call.
   * 
   *
   * Args:
   *   a: compact 2D array of size m x n
   *   b: comapct 2D array of size n x p
   *   out: compact 2D array of size m x p to write the output to
   *   M: rows of a / out
   *   N: columns of a / rows of b
   *   P: columns of b / out
   */

  /// BEGIN YOUR SOLUTION
  dim3 block((L+V-1)/V, (L+V-1)/V);
  dim3 grid((M+L-1)/L, (P+L-1)/L);
  MatmulKernel<<<grid, block>>>(a.ptr, b.ptr, out->ptr, M, N, P);
  
  /// END YOUR SOLUTION
}

////////////////////////////////////////////////////////////////////////////////
// Max and sum reductions
////////////////////////////////////////////////////////////////////////////////

// (old version) 实现简单，不考虑全局内存往block里拉 共享内存。
__global__ void ReduceMaxKernelSimple(const scalar_t* a, scalar_t* out, size_t out_size, size_t reduce_size) {

  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  if (gid < out_size) {
      size_t offset = gid*reduce_size;

    scalar_t maxv = a[offset+0];
    for(int i=1; i< reduce_size; i++) {
      maxv = a[offset+i]> maxv? a[offset+i] : maxv;
    }
    out[gid] = maxv;
  }
}

__global__ void ReduceMaxKernel(const scalar_t* a, scalar_t* out, size_t a_size) {
    // 使用共享内存 --祈祷是一个稍微快一点的版本:）

    extern __shared__ float shared_data[];

    // 计算线程在数组中的全局索引
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 每个线程负责处理一个元素，将数据从全局内存复制到共享内存中
    if (global_idx < a_size) {
        shared_data[threadIdx.x] = a[global_idx];
    }
    // } else {
    //     // 如果线程在数组大小范围之外，则用一个很小的值填充
    //     shared_data[threadIdx.x] = -1e10; // 随机性参数会导致test错误
    // }

    __syncthreads();

    // 通过逐步减半的方式进行归约操作
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x <= stride && (threadIdx.x+stride) < blockDim.x) {
              // 每个线程对比它自己和距离它stride步长的线程的值
              shared_data[threadIdx.x] = fmaxf(shared_data[threadIdx.x], shared_data[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    // 在每个线程块中，第0、1个线程保存了该块的最大值，将其写回全局内存
    if (threadIdx.x == 0) {
        out[blockIdx.x] = fmaxf(shared_data[0], shared_data[1]);
    }
}

void ReduceMax(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  /**
   * Reduce by taking maximum over `reduce_size` contiguous blocks.  Even though it is inefficient,
   * for simplicity you can perform each reduction in a single CUDA thread.
   * 
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over
   */
  /// BEGIN YOUR SOLUTION
  CudaDims dim; 
  dim.block = dim3(reduce_size,1,1);
  dim.grid = dim3(out->size,1,1);

  ReduceMaxKernel<<<dim.grid, dim.block, reduce_size * sizeof(scalar_t)>>>(a.ptr, out->ptr, a.size);
  /// END YOUR SOLUTION

  /// BEGIN YOUR SOLUTION (old version)
  // CudaDims dim = CudaOneDim(out->size);
  // ReduceSumKernelSimple<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, reduce_size);
  /// END YOUR SOLUTION
}


// (old version) 简单版本
__global__ void ReduceSumKernelSimple(const scalar_t* a, scalar_t* out, size_t out_size, size_t reduce_size) {

  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  if (gid < out_size) {
      size_t offset = gid*reduce_size;

    scalar_t sumv = 0;
    for(int i=0; i< reduce_size; i++) {
      sumv += a[offset+i];
    }
    out[gid] = sumv;
  }

}

__global__ void ReduceSumKernel(const scalar_t* a, scalar_t* out, size_t a_size) {
    // 使用共享内存 （好像并没有优化的必要）
    extern __shared__ float shared_data[];

    // 计算线程在数组中的全局索引
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 每个线程负责处理一个元素，将数据从全局内存复制到共享内存中
    if (global_idx < a_size) {
      shared_data[threadIdx.x] = a[global_idx];
    } else {
      shared_data[threadIdx.x] = 0; // 结尾空值项
    }

    __syncthreads();

    if (threadIdx.x == 0) {
      float sum_res = 0;
      // block内部求和shared_data即可
      for (int sid=0; sid<blockDim.x; sid++) {
        sum_res += shared_data[sid];
      }
      out[blockIdx.x] = sum_res;
    }
}

void ReduceSum(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  /**
   * Reduce by taking summation over `reduce_size` contiguous blocks.  Again, for simplicity you 
   * can perform each reduction in a single CUDA thread.
   * 
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over
   */

  /// BEGIN YOUR SOLUTION
  CudaDims dim; 
  dim.block = dim3(reduce_size,1,1);
  dim.grid = dim3(out->size,1,1);

  ReduceSumKernel<<<dim.grid, dim.block, reduce_size * sizeof(scalar_t)>>>(a.ptr, out->ptr, a.size);
  /// END YOUR SOLUTION

  /// BEGIN YOUR SOLUTION (old version)
  // CudaDims dim = CudaOneDim(out->size);
  // ReduceSumKernelSimple<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, reduce_size);
  
  /// END YOUR SOLUTION
}


}  // namespace cuda
}  // namespace needle

PYBIND11_MODULE(ndarray_backend_cuda, m) {
  namespace py = pybind11;
  using namespace needle;
  using namespace cuda;

  m.attr("__device_name__") = "cuda";
  m.attr("__tile_size__") = V;
  // m.attr("__tile_size__") = TILE;

  py::class_<CudaArray>(m, "Array")
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      .def_readonly("size", &CudaArray::size)
      .def("ptr", &CudaArray::ptr_as_int);

  // return numpy array, copying from CPU
  m.def("to_numpy", [](const CudaArray& a, std::vector<size_t> shape, std::vector<size_t> strides,
                       size_t offset) {
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                   [](size_t& c) { return c * ELEM_SIZE; });

    // copy memory to host
    scalar_t* host_ptr = (scalar_t*)std::malloc(a.size * ELEM_SIZE);
    if (host_ptr == 0) throw std::bad_alloc();
    cudaError_t err = cudaMemcpy(host_ptr, a.ptr, a.size * ELEM_SIZE, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));

    // return numpy array
    py::capsule deallocate_buffer(host_ptr, [](void* p) { free(p); });
    return py::array_t<scalar_t>(shape, numpy_strides, host_ptr + offset, deallocate_buffer);
  });

  // copy numpy array to GPU
  m.def("from_numpy", [](py::array_t<scalar_t> a, CudaArray* out) {
    cudaError_t err =
        cudaMemcpy(out->ptr, a.request().ptr, out->size * ELEM_SIZE, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
  });

  m.def("fill", Fill);
  m.def("compact", Compact);
  m.def("ewise_setitem", EwiseSetitem);
  m.def("scalar_setitem", ScalarSetitem);
  m.def("ewise_add", EwiseAdd);
  m.def("scalar_add", ScalarAdd);

  m.def("ewise_mul", EwiseMul);
  m.def("scalar_mul", ScalarMul);
  m.def("ewise_div", EwiseDiv);
  m.def("scalar_div", ScalarDiv);
  m.def("scalar_power", ScalarPower);

  m.def("ewise_maximum", EwiseMaximum);
  m.def("scalar_maximum", ScalarMaximum);
  m.def("ewise_eq", EwiseEq);
  m.def("scalar_eq", ScalarEq);
  m.def("ewise_ge", EwiseGe);
  m.def("scalar_ge", ScalarGe);

  m.def("ewise_log", EwiseLog);
  m.def("ewise_exp", EwiseExp);
  m.def("ewise_tanh", EwiseTanh);

  m.def("matmul", Matmul);

  m.def("reduce_max", ReduceMax);
  m.def("reduce_sum", ReduceSum);
}