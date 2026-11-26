#include <type_traits>

#include "mol_epilogue_kernel.hpp"
#include "nanobind/nanobind.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

namespace nb = nanobind;

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    mol_epilogue_handler,
    mol_epilogue,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::DataType::BF16>>()
        .Arg<ffi::Buffer<ffi::DataType::F32>>()
        .Arg<ffi::Buffer<ffi::DataType::F32>>()
        .Arg<ffi::Buffer<ffi::DataType::F32>>()
        .Arg<ffi::Buffer<ffi::DataType::F32>>()
        .Arg<ffi::Buffer<ffi::DataType::F32>>()
        .Arg<ffi::Buffer<ffi::DataType::F32>>()
        .Arg<ffi::Buffer<ffi::DataType::F32>>()
        .Ret<ffi::Buffer<ffi::DataType::F32>>()
        .Attr<int64_t>("num_users")
        .Attr<int64_t>("num_components")
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    mol_fused_int8_handler,
    mol_fused_int8,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::DataType::S8>>()
        .Arg<ffi::Buffer<ffi::DataType::F32>>()
        .Arg<ffi::Buffer<ffi::DataType::S8>>()
        .Arg<ffi::Buffer<ffi::DataType::F32>>()
        .Arg<ffi::Buffer<ffi::DataType::F32>>()
        .Arg<ffi::Buffer<ffi::DataType::F32>>()
        .Arg<ffi::Buffer<ffi::DataType::F32>>()
        .Arg<ffi::Buffer<ffi::DataType::F32>>()
        .Arg<ffi::Buffer<ffi::DataType::F32>>()
        .Arg<ffi::Buffer<ffi::DataType::F32>>()
        .Arg<ffi::Buffer<ffi::DataType::F32>>()
        .Ret<ffi::Buffer<ffi::DataType::BF16>>()
        .Attr<int64_t>("num_users")
        .Attr<int64_t>("col_offset")
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    mol_gate_bwd_handler,
    mol_gate_bwd,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::DataType::BF16>>()
        .Arg<ffi::Buffer<ffi::DataType::F32>>()
        .Arg<ffi::Buffer<ffi::DataType::F32>>()
        .Arg<ffi::Buffer<ffi::DataType::F32>>()
        .Arg<ffi::Buffer<ffi::DataType::F32>>()
        .Arg<ffi::Buffer<ffi::DataType::F32>>()
        .Arg<ffi::Buffer<ffi::DataType::F32>>()
        .Arg<ffi::Buffer<ffi::DataType::F32>>()
        .Ret<ffi::Buffer<ffi::DataType::BF16>>()
        .Ret<ffi::Buffer<ffi::DataType::F32>>()
        .Ret<ffi::Buffer<ffi::DataType::F32>>()
        .Ret<ffi::Buffer<ffi::DataType::F32>>()
        .Attr<int64_t>("num_users")
);

template <typename T>
nb::capsule encapsulate_ffi_call(T* fn) {
  static_assert(
      std::is_invocable_r_v<XLA_FFI_Error*, T, XLA_FFI_CallFrame*>,
      "Encapsulated function must be an XLA FFI handler"
  );
  return nb::capsule(reinterpret_cast<void*>(fn));
}

NB_MODULE(mol_epilogue_api, m) {
  m.doc() = "Mixture-of-logits retrieval scoring epilogue (bf16 dots -> f32 scores)";
  m.def("mol_epilogue", []() { return encapsulate_ffi_call(mol_epilogue_handler); });
  m.def("mol_fused_int8", []() { return encapsulate_ffi_call(mol_fused_int8_handler); });
  m.attr("mol_fused_int8_col_offset") = true;
  m.def("mol_gate_bwd", []() { return encapsulate_ffi_call(mol_gate_bwd_handler); });
  m.attr("mol_gate_bwd_items_per_block") = kMolGateBwdItemsPerBlock;
  m.attr("mol_gate_bwd_max_users") = kMolGateBwdMaxUsers;
}
