#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace needle {
namespace cpu {

#define ALIGNMENT 256
#define TILE 8
typedef float scalar_t;
const size_t ELEM_SIZE = sizeof(scalar_t);


/**
 * This is a utility structure for maintaining an array aligned to ALIGNMENT boundaries in
 * memory.  This alignment should be at least TILE * ELEM_SIZE, though we make it even larger
 * here by default.
 */
struct AlignedArray {
  AlignedArray(const size_t size) {
    int ret = posix_memalign((void**)&ptr, ALIGNMENT, size * ELEM_SIZE);
    if (ret != 0) throw std::bad_alloc();
    this->size = size;
  }
  ~AlignedArray() { free(ptr); }
  size_t ptr_as_int() {return (size_t)ptr; }
  scalar_t* ptr;
  size_t size;
};



void Fill(AlignedArray* out, scalar_t val) {
  /**
   * Fill the values of an aligned array with val
   */
  for (int i = 0; i < out->size; i++) {
    out->ptr[i] = val;
  }
}


void update_indexing_strategy(std::vector<int32_t>& shape,
                size_t ndim,
                std::vector<int32_t>* p_reversed_base){
  std::vector<int32_t>& reversed_base = *p_reversed_base;
  
  for(size_t i=0; i<ndim; i++){
    // 达到最shape[i]高位，continue进位，并reset当前位 
    if(reversed_base[i]==(shape[ndim-i-1])-1){
      reversed_base[i]=0;
      continue;
    }
    // 当前位置合理+1，成功后直接跳出循环
    reversed_base[i] += 1;
    break;
  }
}



void Compact(const AlignedArray& a, AlignedArray* out, std::vector<int32_t> shape,
             std::vector<int32_t> strides, size_t offset) {
  /**
   * Compact an array in memory
   * 
   * Args:
   *   a: non-compact represntation of the array, given as input
   *   out: compact version of the array to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *a* array (not out, which has compact strides)
   *   offset: offset of the *a* array (not out, which has zero offset, being compact)
   * 
   * Returns:
   *  void (you need to modify out directly, rather than returning anything; this is true for all the
   *  function will implement here, so we won't repeat this note.)
   */
  /// BEGIN YOUR SOLUTION
  
  // 逆序基
  //例：out[strides[0]*i + strides[1]*j + strides[2]*k]]
  // (i,j,k)为strides的基， reverse_base为逆序（k,j,i）
  std::vector<int32_t> reversed_base(shape.size(), 0); 
  
  size_t ndim = shape.size();
  size_t max_item_size = 1;
  for(auto & x : shape) {
    max_item_size *= x;
  }

  for(size_t cnt=0; cnt<max_item_size;) {
    //calc position offset
    size_t positon = 0;
    for(size_t i=0; i<ndim; i++){
      positon += strides[i]*reversed_base[ndim-i-1];
    }
    // set value
    out->ptr[cnt++] = a.ptr[offset+positon];
    // update indexing base
    if(cnt < max_item_size) {
      update_indexing_strategy(shape, ndim, &reversed_base);
    }
  }
  return;
  
  /// END YOUR SOLUTION
}

void EwiseSetitem(const AlignedArray& a, AlignedArray* out, std::vector<int32_t> shape,
                  std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items in a (non-compact) array
   * 
   * Args:
   *   a: _compact_ array whose items will be written to out
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *out* array (not a, which has compact strides)
   *   offset: offset of the *out* array (not a, which has zero offset, being compact)
   */
  /// BEGIN YOUR SOLUTION
  std::vector<int32_t> reversed_base(shape.size(), 0); 
  
  size_t ndim = shape.size();
  size_t max_item_size = 1;
  for(auto & x : shape){
    max_item_size *= x;
  }

  for(size_t cnt=0; cnt<max_item_size;) {
    // 计算需要更新值在out中的position
    size_t positon = 0;
    for(size_t i=0; i<ndim; i++){
      positon += strides[i]*reversed_base[ndim-i-1];
    }

    // setitem (from compact a to non-compact out)
    out->ptr[offset+positon] = a.ptr[cnt++];
    
    if(cnt < max_item_size) {
      update_indexing_strategy(shape, ndim, &reversed_base);
    }
  }
  return;
  /// END YOUR SOLUTION
}

void ScalarSetitem(const size_t size, scalar_t val, AlignedArray* out, std::vector<int32_t> shape,
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
  std::vector<int32_t> reversed_base(shape.size(), 0); 
  
  size_t ndim = shape.size();

  for(size_t cnt=0; cnt<size;) {
    // 计算需要更新值在out中的position
    size_t positon = 0;
    for(size_t i=0; i<ndim; i++){
      positon += strides[i]*reversed_base[ndim-i-1];
    }

    // setitem (from compact a to non-compact out)
    out->ptr[offset+positon] = val;
    
    cnt++;
    if(cnt < size) {
      update_indexing_strategy(shape, ndim, &reversed_base);
    }
  }
  return;
  
  /// END YOUR SOLUTION
}

void EwiseAdd(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  /**
   * Set entries in out to be the sum of correspondings entires in a and b.
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] + b.ptr[i];
  }
}

void ScalarAdd(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  /**
   * Set entries in out to be the sum of correspondings entry in a plus the scalar val.
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] + val;
  }
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
#define EWISE_OP(name, op) \
void Ewise##name(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) { \
  for (size_t i = 0; i < a.size; i++) { \
    out->ptr[i] = a.ptr[i] op b.ptr[i]; \
  } \
}

#define SCALAR_OP(name, op) \
void Scalar##name(const AlignedArray& a, scalar_t val, AlignedArray* out) { \
  for (size_t i = 0; i < a.size; i++) { \
    out->ptr[i] = a.ptr[i] op val; \
  } \
}

#define EWISE_UFUNC(name, func) \
void Ewise##name(const AlignedArray& a, AlignedArray* out) { \
  for (size_t i = 0; i < a.size; i++) { \
    out->ptr[i] = func(a.ptr[i]); \
  } \
}

#define EWISE_BFUNC(name, func) \
void Ewise##name(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) { \
  for (size_t i = 0; i < a.size; i++) { \
    out->ptr[i] = func(a.ptr[i], b.ptr[i]); \
  } \
}

#define SCALAR_BFUNC(name, func) \
void Scalar##name(const AlignedArray& a, scalar_t val, AlignedArray* out) { \
  for (size_t i = 0; i < a.size; i++) { \
    out->ptr[i] = func(a.ptr[i], val); \
  } \
}

EWISE_OP(Mul, *);
EWISE_OP(Div, /);
EWISE_OP(Eq, ==);
EWISE_OP(Ge, >=);
EWISE_BFUNC(Maximum, [](scalar_t a, scalar_t b) { return (a > b) ? a : b; });
EWISE_UFUNC(Log, log);
EWISE_UFUNC(Exp, exp);
EWISE_UFUNC(Tanh, tanh);
SCALAR_OP(Mul, *);
SCALAR_OP(Div, /);
SCALAR_OP(Eq, ==);
SCALAR_OP(Ge, >=);
SCALAR_BFUNC(Power, pow);
SCALAR_BFUNC(Maximum, [](scalar_t a, scalar_t b) { return (a > b) ? a : b; });

/// END YOUR SOLUTION

/// BEGIN YOUR SOLUTION

// c++ 模板解法，不如宏方便。

// namespace tp {
//   template<typename Op>
//   void EwiseTemp(const AlignedArray &a, const AlignedArray &b, AlignedArray *out) {
//       Op o;
//       for (size_t i = 0; i < a.size; i++) {
//           out->ptr[i] = o(a.ptr[i], b.ptr[i]);
//       }
//   };

//   template<typename Op>
//   void ScalarTemp(const AlignedArray &a, scalar_t val, AlignedArray *out) {
//       Op o;
//       for (size_t i = 0; i < a.size; i++) {
//           out->ptr[i] = o(a.ptr[i], val);
//       }
//   };

//   template<typename Op>
//   void EwiseSingleOpTemp(const AlignedArray &a, AlignedArray *out) {
//       Op o;
//       for (size_t i = 0; i < out->size; i++) {
//           out->ptr[i] = o(a.ptr[i]);
//       }
//   };

//   template <typename T>
//   struct max {
//       constexpr const T& operator()(const T& lhs, const T& rhs) const {
//           return lhs > rhs ? lhs : rhs;
//       }
//       using is_transparent = void;
//   };

//   template <typename T>
//   struct pow {
//       constexpr T operator()(const T& base, const T& exponent) const {
//           // return std::pow(base, exponent);
//           return std::pow(base, exponent);
//       }
//       using is_transparent = void;
//   };

//   template <typename T>
//   struct log {
//       constexpr T operator()(const T& x) const {
//           return std::log(x);
//       }
//       using is_transparent = void;
//   };

//   template <typename T>
//   struct exp {
//       constexpr T operator()(const T& x) const {
//           return std::exp(x);
//       }
//       using is_transparent = void;
//   };

//   template <typename T>
//   struct tanh {
//       constexpr T operator()(const T& x) const {
//           return std::tanh(x);
//       }
//       using is_transparent = void;
//   };
// }

// auto EwiseMul = tp::EwiseTemp<std::multiplies<scalar_t>>;
// auto ScalarMul = tp::ScalarTemp<std::multiplies<scalar_t>>;

// auto EwiseDiv = tp::EwiseTemp<std::divides<scalar_t>>;
// auto ScalarDiv = tp::ScalarTemp<std::divides<scalar_t>>;

// auto EwiseEq = tp::EwiseTemp<std::equal_to<scalar_t>>;
// auto ScalarEq = tp::ScalarTemp<std::equal_to<scalar_t>>;

// auto EwiseGe = tp::EwiseTemp<std::greater_equal<scalar_t>>;
// auto ScalarGe = tp::ScalarTemp<std::greater_equal<scalar_t>>;

// auto ScalarPower = tp::ScalarTemp<tp::pow<scalar_t>>;
// auto ScalarMaximum = tp::ScalarTemp<tp::max<scalar_t>>;
// auto EwiseMaximum = tp::EwiseTemp<tp::max<scalar_t>>;

// auto EwiseLog = tp::EwiseSingleOpTemp<tp::log<scalar_t>>;
// auto EwiseExp = tp::EwiseSingleOpTemp<tp::exp<scalar_t>>;
// auto EwiseTanh = tp::EwiseSingleOpTemp<tp::tanh<scalar_t>>;
/// END YOUR SOLUTION


void Matmul(const AlignedArray& a, const AlignedArray& b, AlignedArray* out, uint32_t m, uint32_t n,
            uint32_t p) {
  /**
   * Multiply two (compact) matrices into an output (also comapct) matrix.  For this implementation
   * you can use the "naive" three-loop algorithm.
   *
   * Args:
   *   a: compact 2D array of size m x n
   *   b: comapct 2D array of size n x p
   *   out: compact 2D array of size m x p to write the output to
   *   m: rows of a / out
   *   n: columns of a / rows of b
   *   p: coolumns of b / out
   */

  /// BEGIN YOUR SOLUTION
  for(size_t i=0; i<m; i++) {
    for (size_t j=0; j<p; j++) {
      scalar_t sumv = 0;
      for (size_t k=0; k<n; k++) {
        sumv += a.ptr[i*n+k] * b.ptr[k*p+j];
      }
      out->ptr[i*p+j] = sumv;
    }
  }
  
  /// END YOUR SOLUTION
}

inline void AlignedDot(const float* __restrict__ a, 
                       const float* __restrict__ b, 
                       float* __restrict__ out) {

  /**
   * Multiply together two TILE x TILE matrices, and _add _the result to out (it is important to add
   * the result to the existing out, which you should not set to zero beforehand).  We are including
   * the compiler flags here that enable the compile to properly use vector operators to implement 
   * this function.  Specifically, the __restrict__ keyword indicates to the compile that a, b, and 
   * out don't have any overlapping memory (which is necessary in order for vector operations to be 
   * equivalent to their non-vectorized counterparts (imagine what could happen otherwise if a, b, 
   * and out had overlapping memory).  Similarly the __builtin_assume_aligned keyword tells the 
   * compiler that the input array siwll be aligned to the appropriate blocks in memory, which also 
   * helps the compiler vectorize the code.
   *
   * Args:
   *   a: compact 2D array of size TILE x TILE
   *   b: compact 2D array of size TILE x TILE
   *   out: compact 2D array of size TILE x TILE to write to
   */

  a = (const float*)__builtin_assume_aligned(a, TILE * ELEM_SIZE);
  b = (const float*)__builtin_assume_aligned(b, TILE * ELEM_SIZE);
  out = (float*)__builtin_assume_aligned(out, TILE * ELEM_SIZE);

  /// BEGIN YOUR SOLUTION
  for (size_t i = 0; i < TILE; i++) {
    for (size_t j = 0; j < TILE; j++) {
      for (size_t k = 0; k < TILE; k++) {
        out[i * TILE + j] += a[i * TILE + k] * b[k * TILE + j];
      }
    }
  }
  /// END YOUR SOLUTION
}

void MatmulTiled(const AlignedArray& a, const AlignedArray& b, AlignedArray* out, uint32_t m,
                 uint32_t n, uint32_t p) {
  /**
   * Matrix multiplication on tiled representations of array.  In this setting, a, b, and out
   * are all *4D* compact arrays of the appropriate size, e.g. a is an array of size
   *   a[m/TILE][n/TILE][TILE][TILE]
   * You should do the multiplication tile-by-tile to improve performance of the array (i.e., this
   * function should call `AlignedDot()` implemented above).
   * 
   * Note that this function will only be called when m, n, p are all multiples of TILE, so you can
   * assume that this division happens without any remainder.
   * 
   * Args:
   *   a: compact 4D array of size m/TILE x n/TILE x TILE x TILE
   *   b: compact 4D array of size n/TILE x p/TILE x TILE x TILE
   *   out: compact 4D array of size m/TILE x p/TILE x TILE x TILE to write to
   *   m: rows of a / out
   *   n: columns of a / rows of b
   *   p: coolumns of b / out
   * 
   */
  /// BEGIN YOUR SOLUTION
    // set out=0
    memset(out->ptr, 0, out->size* sizeof(scalar_t));
    // for (size_t i = 0; i<out->size; i++) {
    //   out->ptr[i] = 0;
    // }

    for (size_t i=0; i<m/TILE; i++) {
      for (size_t j=0; j<p/TILE; j++){
        scalar_t* p_out = out->ptr + (i*p/TILE+j)*TILE*TILE;
        for(size_t k=0; k<n/TILE; k++) {
          const scalar_t* p_a = a.ptr + (i*n/TILE+k)*TILE*TILE;
          const scalar_t* p_b = b.ptr + (k*p/TILE+j)*TILE*TILE;
          
          AlignedDot(p_a, p_b, p_out);
        }
      }
    }
  
  /// END YOUR SOLUTION
}

void ReduceMax(const AlignedArray& a, AlignedArray* out, size_t reduce_size) {
  /**
   * Reduce by taking maximum over `reduce_size` contiguous blocks.
   * 
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over
   */

  /// BEGIN YOUR SOLUTION
  for(size_t i=0; i<out->size;i++) {
    size_t offset = i*reduce_size;
    
    scalar_t maxv = a.ptr[offset+0]; 
    for (size_t j=1;j<reduce_size;j++) {
      maxv = std::max(maxv, a.ptr[offset+j]); 
    }
    out->ptr[i] = maxv;
  }
  
  /// END YOUR SOLUTION
}

void ReduceSum(const AlignedArray& a, AlignedArray* out, size_t reduce_size) {
  /**
   * Reduce by taking sum over `reduce_size` contiguous blocks.
   * 
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over
   */

  /// BEGIN YOUR SOLUTION
  for(size_t i=0; i<out->size; i++) {
    size_t offset = i*reduce_size;
    
    scalar_t sumv = 0;
    for (size_t j=0; j<reduce_size; j++) {
      sumv += a.ptr[offset+j];
    }
    out->ptr[i] = sumv;
  }
  
  
  /// END YOUR SOLUTION
}

}  // namespace cpu
}  // namespace needle

PYBIND11_MODULE(ndarray_backend_cpu, m) {
  namespace py = pybind11;
  using namespace needle;
  using namespace cpu;

  m.attr("__device_name__") = "cpu";
  m.attr("__tile_size__") = TILE;

  py::class_<AlignedArray>(m, "Array")
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      .def("ptr", &AlignedArray::ptr_as_int)
      .def_readonly("size", &AlignedArray::size);

  // return numpy array (with copying for simplicity, otherwise garbage
  // collection is a pain)
  m.def("to_numpy", [](const AlignedArray& a, std::vector<size_t> shape,
                       std::vector<size_t> strides, size_t offset) {
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                   [](size_t& c) { return c * ELEM_SIZE; });
    return py::array_t<scalar_t>(shape, numpy_strides, a.ptr + offset);
  });

  // convert from numpy (with copying)
  m.def("from_numpy", [](py::array_t<scalar_t> a, AlignedArray* out) {
    std::memcpy(out->ptr, a.request().ptr, out->size * ELEM_SIZE);
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
  m.def("matmul_tiled", MatmulTiled);

  m.def("reduce_max", ReduceMax);
  m.def("reduce_sum", ReduceSum);
}
