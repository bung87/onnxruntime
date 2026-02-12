## ort_bindings.nim
## Low-level Nim bindings for ONNX Runtime C API

import std/[os, strutils]

const
  ORT_API_VERSION* = 24

# Detect ONNX Runtime installation path at compile time
const OnnxRuntimePath = block:
  var path = ""
  when defined(macosx):
    let brewPrefix = strip(staticExec("brew --prefix onnxruntime 2>/dev/null || echo ''"))
    if brewPrefix != "":
      path = brewPrefix
    else:
      for p in ["/opt/homebrew/opt/onnxruntime", "/usr/local/opt/onnxruntime"]:
        if dirExists(p):
          path = p
          break
  elif defined(linux):
    for p in ["/usr/local", "/usr", "/opt/onnxruntime"]:
      if fileExists(p / "lib/libonnxruntime.so") or dirExists(p / "include/onnxruntime"):
        path = p
        break
  path

# Platform-specific library path
when defined(macosx):
  const OrtLibName* = 
    when OnnxRuntimePath != "": OnnxRuntimePath / "lib/libonnxruntime.dylib"
    else: "libonnxruntime.dylib"
elif defined(windows):
  const OrtLibName* = "onnxruntime.dll"
else:
  const OrtLibName* = 
    when OnnxRuntimePath != "": OnnxRuntimePath / "lib/libonnxruntime.so"
    else: "libonnxruntime.so"

{.push, header: "<onnxruntime/onnxruntime_c_api.h>", cdecl.}

type
  ONNXTensorElementDataType* {.size: sizeof(cint).} = enum
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING
    ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128
    ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FNUZ
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2FNUZ
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT4
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4

  OrtLoggingLevel* {.size: sizeof(cint).} = enum
    ORT_LOGGING_LEVEL_VERBOSE
    ORT_LOGGING_LEVEL_INFO
    ORT_LOGGING_LEVEL_WARNING
    ORT_LOGGING_LEVEL_ERROR
    ORT_LOGGING_LEVEL_FATAL

  OrtErrorCode* {.size: sizeof(cint).} = enum
    ORT_OK
    ORT_FAIL
    ORT_INVALID_ARGUMENT
    ORT_NO_SUCHFILE
    ORT_NO_MODEL
    ORT_ENGINE_ERROR
    ORT_RUNTIME_EXCEPTION
    ORT_INVALID_PROTOBUF
    ORT_MODEL_LOADED
    ORT_NOT_IMPLEMENTED
    ORT_INVALID_GRAPH
    ORT_EP_FAIL

  GraphOptimizationLevel* {.size: sizeof(cint).} = enum
    ORT_DISABLE_ALL = 0
    ORT_ENABLE_BASIC = 1
    ORT_ENABLE_EXTENDED = 2
    ORT_ENABLE_ALL = 99

  ExecutionMode* {.size: sizeof(cint).} = enum
    ORT_SEQUENTIAL = 0
    ORT_PARALLEL = 1

  OrtAllocatorType* {.size: sizeof(cint).} = enum
    OrtInvalidAllocator = -1
    OrtDeviceAllocator = 0
    OrtArenaAllocator = 1

  OrtMemType* {.size: sizeof(cint).} = enum
    OrtMemTypeCPUInput = -2
    OrtMemTypeCPUOutput = -1
    OrtMemTypeDefault = 0

  # Opaque types
  OrtEnvObj = object
  OrtEnv* = ptr OrtEnvObj
  
  OrtStatusObj = object
  OrtStatus* = ptr OrtStatusObj
  OrtStatusPtr* = OrtStatus
  
  OrtMemoryInfoObj = object
  OrtMemoryInfo* = ptr OrtMemoryInfoObj
  
  OrtSessionObj = object
  OrtSession* = ptr OrtSessionObj
  
  OrtValueObj = object
  OrtValue* = ptr OrtValueObj
  
  OrtRunOptionsObj = object
  OrtRunOptions* = ptr OrtRunOptionsObj
  
  OrtTypeInfoObj = object
  OrtTypeInfo* = ptr OrtTypeInfoObj
  
  OrtTensorTypeAndShapeInfoObj = object
  OrtTensorTypeAndShapeInfo* = ptr OrtTensorTypeAndShapeInfoObj
  
  OrtSessionOptionsObj = object
  OrtSessionOptions* = ptr OrtSessionOptionsObj
  
  OrtAllocatorObj = object
  OrtAllocator* = ptr OrtAllocatorObj

{.pop.}

# Use emit to create a C wrapper that handles all the API details
{.emit: """
#include <onnxruntime/onnxruntime_c_api.h>
#include <stdlib.h>

static const OrtApi* g_ort_api = NULL;

static int ensure_api_initialized(void) {
    if (g_ort_api == NULL) {
        const OrtApiBase* base = OrtGetApiBase();
        if (base == NULL) return -1;
        g_ort_api = base->GetApi(24);  // ORT_API_VERSION
    }
    return (g_ort_api != NULL) ? 0 : -1;
}

// Status wrappers
const char* ort_GetErrorMessage(OrtStatus* status) {
    if (!g_ort_api) return NULL;
    return g_ort_api->GetErrorMessage(status);
}

void ort_ReleaseStatus(OrtStatus* status) {
    if (!g_ort_api) return;
    g_ort_api->ReleaseStatus(status);
}

// Environment wrappers
OrtStatus* ort_CreateEnv(OrtLoggingLevel log_level, const char* logid, OrtEnv** out) {
    if (ensure_api_initialized() != 0) return NULL;
    return g_ort_api->CreateEnv(log_level, logid, out);
}

void ort_ReleaseEnv(OrtEnv* env) {
    if (!g_ort_api || !env) return;
    g_ort_api->ReleaseEnv(env);
}

// Session wrappers
OrtStatus* ort_CreateSession(OrtEnv* env, const char* model_path, OrtSessionOptions* options, OrtSession** out) {
    if (ensure_api_initialized() != 0) return NULL;
    return g_ort_api->CreateSession(env, model_path, options, out);
}

void ort_ReleaseSession(OrtSession* session) {
    if (!g_ort_api || !session) return;
    g_ort_api->ReleaseSession(session);
}

OrtStatus* ort_Run(OrtSession* session, OrtRunOptions* run_options,
                   const char* const* input_names, const OrtValue* const* inputs, size_t input_len,
                   const char* const* output_names, size_t output_names_len, OrtValue** outputs) {
    if (ensure_api_initialized() != 0) return NULL;
    return g_ort_api->Run(session, run_options, input_names, inputs, input_len, 
                          output_names, output_names_len, outputs);
}

// SessionOptions wrappers
OrtStatus* ort_CreateSessionOptions(OrtSessionOptions** options) {
    if (ensure_api_initialized() != 0) return NULL;
    return g_ort_api->CreateSessionOptions(options);
}

void ort_ReleaseSessionOptions(OrtSessionOptions* options) {
    if (!g_ort_api || !options) return;
    g_ort_api->ReleaseSessionOptions(options);
}

OrtStatus* ort_EnableCpuMemArena(OrtSessionOptions* options) {
    if (ensure_api_initialized() != 0) return NULL;
    return g_ort_api->EnableCpuMemArena(options);
}

// MemoryInfo wrappers
OrtStatus* ort_CreateCpuMemoryInfo(OrtAllocatorType type, OrtMemType mem_type, OrtMemoryInfo** out) {
    if (ensure_api_initialized() != 0) return NULL;
    return g_ort_api->CreateCpuMemoryInfo(type, mem_type, out);
}

void ort_ReleaseMemoryInfo(OrtMemoryInfo* info) {
    if (!g_ort_api || !info) return;
    g_ort_api->ReleaseMemoryInfo(info);
}

// Allocator wrappers
OrtStatus* ort_GetAllocatorWithDefaultOptions(OrtAllocator** out) {
    if (ensure_api_initialized() != 0) return NULL;
    return g_ort_api->GetAllocatorWithDefaultOptions(out);
}

void ort_ReleaseAllocator(OrtAllocator* allocator) {
    if (!g_ort_api || !allocator) return;
    g_ort_api->ReleaseAllocator(allocator);
}

// Tensor wrappers
OrtStatus* ort_CreateTensorWithDataAsOrtValue(OrtMemoryInfo* info, void* p_data, size_t p_data_len,
                                               int64_t* shape, size_t shape_len,
                                               ONNXTensorElementDataType type, OrtValue** out) {
    if (ensure_api_initialized() != 0) return NULL;
    return g_ort_api->CreateTensorWithDataAsOrtValue(info, p_data, p_data_len, shape, shape_len, 
                                                      type, out);
}

void ort_ReleaseValue(OrtValue* value) {
    if (!g_ort_api || !value) return;
    g_ort_api->ReleaseValue(value);
}

OrtStatus* ort_GetTensorMutableData(OrtValue* value, void** out) {
    if (ensure_api_initialized() != 0) return NULL;
    return g_ort_api->GetTensorMutableData(value, out);
}

OrtStatus* ort_GetTensorTypeAndShape(OrtValue* value, OrtTensorTypeAndShapeInfo** out) {
    if (ensure_api_initialized() != 0) return NULL;
    return g_ort_api->GetTensorTypeAndShape(value, out);
}

// TypeInfo wrappers
OrtStatus* ort_GetTypeInfo(OrtValue* value, OrtTypeInfo** out) {
    if (ensure_api_initialized() != 0) return NULL;
    return g_ort_api->GetTypeInfo(value, out);
}

void ort_ReleaseTypeInfo(OrtTypeInfo* type_info) {
    if (!g_ort_api || !type_info) return;
    g_ort_api->ReleaseTypeInfo(type_info);
}

OrtStatus* ort_CastTypeInfoToTensorInfo(OrtTypeInfo* type_info, OrtTensorTypeAndShapeInfo** out) {
    if (ensure_api_initialized() != 0) return NULL;
    return g_ort_api->CastTypeInfoToTensorInfo(type_info, out);
}

// TensorTypeAndShapeInfo wrappers
void ort_ReleaseTensorTypeAndShapeInfo(OrtTensorTypeAndShapeInfo* info) {
    if (!g_ort_api || !info) return;
    g_ort_api->ReleaseTensorTypeAndShapeInfo(info);
}

OrtStatus* ort_GetDimensionsCount(OrtTensorTypeAndShapeInfo* info, size_t* out) {
    if (ensure_api_initialized() != 0) return NULL;
    return g_ort_api->GetDimensionsCount(info, out);
}

OrtStatus* ort_GetDimensions(OrtTensorTypeAndShapeInfo* info, int64_t* dim_values, size_t dim_values_length) {
    if (ensure_api_initialized() != 0) return NULL;
    return g_ort_api->GetDimensions(info, dim_values, dim_values_length);
}

OrtStatus* ort_GetTensorShapeElementCount(OrtTensorTypeAndShapeInfo* info, size_t* out) {
    if (ensure_api_initialized() != 0) return NULL;
    return g_ort_api->GetTensorShapeElementCount(info, out);
}

// Session introspection wrappers
OrtStatus* ort_SessionGetInputCount(OrtSession* session, size_t* out) {
    if (ensure_api_initialized() != 0) return NULL;
    return g_ort_api->SessionGetInputCount(session, out);
}

OrtStatus* ort_SessionGetOutputCount(OrtSession* session, size_t* out) {
    if (ensure_api_initialized() != 0) return NULL;
    return g_ort_api->SessionGetOutputCount(session, out);
}

OrtStatus* ort_SessionGetInputName(OrtSession* session, size_t index, OrtAllocator* allocator, char** out) {
    if (ensure_api_initialized() != 0) return NULL;
    return g_ort_api->SessionGetInputName(session, index, allocator, out);
}

OrtStatus* ort_SessionGetOutputName(OrtSession* session, size_t index, OrtAllocator* allocator, char** out) {
    if (ensure_api_initialized() != 0) return NULL;
    return g_ort_api->SessionGetOutputName(session, index, allocator, out);
}

OrtStatus* ort_SessionGetInputTypeInfo(OrtSession* session, size_t index, OrtTypeInfo** out) {
    if (ensure_api_initialized() != 0) return NULL;
    return g_ort_api->SessionGetInputTypeInfo(session, index, out);
}

OrtStatus* ort_SessionGetOutputTypeInfo(OrtSession* session, size_t index, OrtTypeInfo** out) {
    if (ensure_api_initialized() != 0) return NULL;
    return g_ort_api->SessionGetOutputTypeInfo(session, index, out);
}
""".}

# Import the C wrapper functions
proc ort_GetErrorMessage*(status: OrtStatusPtr): cstring {.importc: "ort_GetErrorMessage", nodecl.}
proc ort_ReleaseStatus*(status: OrtStatusPtr) {.importc: "ort_ReleaseStatus", nodecl.}
proc ort_CreateEnv*(log_level: OrtLoggingLevel, logid: cstring, outs: ptr OrtEnv): OrtStatusPtr {.importc: "ort_CreateEnv", nodecl.}
proc ort_ReleaseEnv*(env: OrtEnv) {.importc: "ort_ReleaseEnv", nodecl.}
proc ort_CreateSession*(env: OrtEnv, model_path: cstring, options: OrtSessionOptions, outs: ptr OrtSession): OrtStatusPtr {.importc: "ort_CreateSession", nodecl.}
proc ort_ReleaseSession*(session: OrtSession) {.importc: "ort_ReleaseSession", nodecl.}
proc ort_Run*(session: OrtSession, run_options: OrtRunOptions,
              input_names: ptr cstring, inputs: ptr OrtValue, input_len: csize_t,
              output_names: ptr cstring, output_names_len: csize_t,
              outputs: ptr OrtValue): OrtStatusPtr {.importc: "ort_Run", nodecl.}
proc ort_CreateSessionOptions*(options: ptr OrtSessionOptions): OrtStatusPtr {.importc: "ort_CreateSessionOptions", nodecl.}
proc ort_ReleaseSessionOptions*(options: OrtSessionOptions) {.importc: "ort_ReleaseSessionOptions", nodecl.}
proc ort_EnableCpuMemArena*(options: OrtSessionOptions): OrtStatusPtr {.importc: "ort_EnableCpuMemArena", nodecl.}
proc ort_CreateCpuMemoryInfo*(`type`: OrtAllocatorType, mem_type: OrtMemType, outs: ptr OrtMemoryInfo): OrtStatusPtr {.importc: "ort_CreateCpuMemoryInfo", nodecl.}
proc ort_ReleaseMemoryInfo*(info: OrtMemoryInfo) {.importc: "ort_ReleaseMemoryInfo", nodecl.}
proc ort_CreateTensorWithDataAsOrtValue*(info: OrtMemoryInfo, p_data: pointer, p_data_len: csize_t,
                                          shape: ptr int64, shape_len: csize_t,
                                          `type`: ONNXTensorElementDataType, outs: ptr OrtValue): OrtStatusPtr {.importc: "ort_CreateTensorWithDataAsOrtValue", nodecl.}
proc ort_ReleaseValue*(value: OrtValue) {.importc: "ort_ReleaseValue", nodecl.}
proc ort_GetTensorMutableData*(value: OrtValue, outs: ptr pointer): OrtStatusPtr {.importc: "ort_GetTensorMutableData", nodecl.}
proc ort_GetTensorTypeAndShape*(value: OrtValue, outs: ptr OrtTensorTypeAndShapeInfo): OrtStatusPtr {.importc: "ort_GetTensorTypeAndShape", nodecl.}
proc ort_GetTypeInfo*(value: OrtValue, outs: ptr OrtTypeInfo): OrtStatusPtr {.importc: "ort_GetTypeInfo", nodecl.}
proc ort_ReleaseTypeInfo*(type_info: OrtTypeInfo) {.importc: "ort_ReleaseTypeInfo", nodecl.}
proc ort_CastTypeInfoToTensorInfo*(type_info: OrtTypeInfo, outs: ptr OrtTensorTypeAndShapeInfo): OrtStatusPtr {.importc: "ort_CastTypeInfoToTensorInfo", nodecl.}
proc ort_ReleaseTensorTypeAndShapeInfo*(info: OrtTensorTypeAndShapeInfo) {.importc: "ort_ReleaseTensorTypeAndShapeInfo", nodecl.}
proc ort_GetDimensionsCount*(info: OrtTensorTypeAndShapeInfo, outs: ptr csize_t): OrtStatusPtr {.importc: "ort_GetDimensionsCount", nodecl.}
proc ort_GetDimensions*(info: OrtTensorTypeAndShapeInfo, dim_values: ptr int64, dim_values_length: csize_t): OrtStatusPtr {.importc: "ort_GetDimensions", nodecl.}
proc ort_GetTensorShapeElementCount*(info: OrtTensorTypeAndShapeInfo, outs: ptr csize_t): OrtStatusPtr {.importc: "ort_GetTensorShapeElementCount", nodecl.}
proc ort_GetAllocatorWithDefaultOptions*(outs: ptr OrtAllocator): OrtStatusPtr {.importc: "ort_GetAllocatorWithDefaultOptions", nodecl.}
proc ort_ReleaseAllocator*(allocator: OrtAllocator) {.importc: "ort_ReleaseAllocator", nodecl.}
proc ort_SessionGetInputCount*(session: OrtSession, outs: ptr csize_t): OrtStatusPtr {.importc: "ort_SessionGetInputCount", nodecl.}
proc ort_SessionGetOutputCount*(session: OrtSession, outs: ptr csize_t): OrtStatusPtr {.importc: "ort_SessionGetOutputCount", nodecl.}
proc ort_SessionGetInputName*(session: OrtSession, index: csize_t, allocator: OrtAllocator, outs: ptr cstring): OrtStatusPtr {.importc: "ort_SessionGetInputName", nodecl.}
proc ort_SessionGetOutputName*(session: OrtSession, index: csize_t, allocator: OrtAllocator, outs: ptr cstring): OrtStatusPtr {.importc: "ort_SessionGetOutputName", nodecl.}
proc ort_SessionGetInputTypeInfo*(session: OrtSession, index: csize_t, outs: ptr OrtTypeInfo): OrtStatusPtr {.importc: "ort_SessionGetInputTypeInfo", nodecl.}
proc ort_SessionGetOutputTypeInfo*(session: OrtSession, index: csize_t, outs: ptr OrtTypeInfo): OrtStatusPtr {.importc: "ort_SessionGetOutputTypeInfo", nodecl.}

# Nim-friendly wrappers (same names as before, but calling the C wrappers)
proc CreateEnv*(log_severity_level: OrtLoggingLevel, logid: cstring, outs: ptr OrtEnv): OrtStatusPtr =
  ort_CreateEnv(log_severity_level, logid, outs)

proc ReleaseEnv*(env: OrtEnv) = ort_ReleaseEnv(env)

proc CreateSession*(env: OrtEnv, model_path: cstring, options: OrtSessionOptions, outs: ptr OrtSession): OrtStatusPtr =
  ort_CreateSession(env, model_path, options, outs)

proc ReleaseSession*(session: OrtSession) = ort_ReleaseSession(session)

proc Run*(session: OrtSession, run_options: OrtRunOptions,
          input_names: ptr cstring, inputs: ptr OrtValue, input_len: csize_t,
          output_names: ptr cstring, output_names_len: csize_t,
          outputs: ptr OrtValue): OrtStatusPtr =
  ort_Run(session, run_options, input_names, inputs, input_len, 
          output_names, output_names_len, outputs)

proc CreateSessionOptions*(options: ptr OrtSessionOptions): OrtStatusPtr =
  ort_CreateSessionOptions(options)

proc ReleaseSessionOptions*(options: OrtSessionOptions) = ort_ReleaseSessionOptions(options)

proc EnableCpuMemArena*(options: OrtSessionOptions): OrtStatusPtr =
  ort_EnableCpuMemArena(options)

proc CreateCpuMemoryInfo*(`type`: OrtAllocatorType, mem_type: OrtMemType, outs: ptr OrtMemoryInfo): OrtStatusPtr =
  ort_CreateCpuMemoryInfo(`type`, mem_type, outs)

proc ReleaseMemoryInfo*(info: OrtMemoryInfo) = ort_ReleaseMemoryInfo(info)

proc GetAllocatorWithDefaultOptions*(outs: ptr OrtAllocator): OrtStatusPtr =
  ort_GetAllocatorWithDefaultOptions(outs)

proc ReleaseAllocator*(allocator: OrtAllocator) =
  ort_ReleaseAllocator(allocator)

# Session introspection APIs
proc SessionGetInputCount*(session: OrtSession, outs: ptr csize_t): OrtStatusPtr =
  ort_SessionGetInputCount(session, outs)

proc SessionGetOutputCount*(session: OrtSession, outs: ptr csize_t): OrtStatusPtr =
  ort_SessionGetOutputCount(session, outs)

proc SessionGetInputName*(session: OrtSession, index: csize_t, allocator: OrtAllocator, outs: ptr cstring): OrtStatusPtr =
  ort_SessionGetInputName(session, index, allocator, outs)

proc SessionGetOutputName*(session: OrtSession, index: csize_t, allocator: OrtAllocator, outs: ptr cstring): OrtStatusPtr =
  ort_SessionGetOutputName(session, index, allocator, outs)

proc SessionGetInputTypeInfo*(session: OrtSession, index: csize_t, outs: ptr OrtTypeInfo): OrtStatusPtr =
  ort_SessionGetInputTypeInfo(session, index, outs)

proc SessionGetOutputTypeInfo*(session: OrtSession, index: csize_t, outs: ptr OrtTypeInfo): OrtStatusPtr =
  ort_SessionGetOutputTypeInfo(session, index, outs)

proc CreateTensorWithDataAsOrtValue*(info: OrtMemoryInfo, p_data: pointer, p_data_len: csize_t,
                                     shape: ptr int64, shape_len: csize_t,
                                     `type`: ONNXTensorElementDataType, outs: ptr OrtValue): OrtStatusPtr =
  ort_CreateTensorWithDataAsOrtValue(info, p_data, p_data_len, shape, shape_len, `type`, outs)

proc ReleaseValue*(value: OrtValue) = ort_ReleaseValue(value)

proc GetTensorMutableData*(value: OrtValue, outs: ptr pointer): OrtStatusPtr =
  ort_GetTensorMutableData(value, outs)

proc GetTensorTypeAndShape*(value: OrtValue, outs: ptr OrtTensorTypeAndShapeInfo): OrtStatusPtr =
  ort_GetTensorTypeAndShape(value, outs)

proc GetTypeInfo*(value: OrtValue, outs: ptr OrtTypeInfo): OrtStatusPtr =
  ort_GetTypeInfo(value, outs)

proc ReleaseTypeInfo*(type_info: OrtTypeInfo) = ort_ReleaseTypeInfo(type_info)

proc CastTypeInfoToTensorInfo*(type_info: OrtTypeInfo, outs: ptr OrtTensorTypeAndShapeInfo): OrtStatusPtr =
  ort_CastTypeInfoToTensorInfo(type_info, outs)

proc ReleaseTensorTypeAndShapeInfo*(info: OrtTensorTypeAndShapeInfo) = ort_ReleaseTensorTypeAndShapeInfo(info)

proc GetDimensionsCount*(info: OrtTensorTypeAndShapeInfo, outs: ptr csize_t): OrtStatusPtr =
  ort_GetDimensionsCount(info, outs)

proc GetDimensions*(info: OrtTensorTypeAndShapeInfo, dim_values: ptr int64, dim_values_length: csize_t): OrtStatusPtr =
  ort_GetDimensions(info, dim_values, dim_values_length)

proc GetTensorShapeElementCount*(info: OrtTensorTypeAndShapeInfo, outs: ptr csize_t): OrtStatusPtr =
  ort_GetTensorShapeElementCount(info, outs)

proc GetErrorMessage*(status: OrtStatusPtr): cstring = ort_GetErrorMessage(status)

proc ReleaseStatus*(status: OrtStatusPtr) = ort_ReleaseStatus(status)

# Helper proc to check status and raise exception on error
proc checkStatus*(status: OrtStatusPtr) =
  if status != nil:
    let msg = $GetErrorMessage(status)
    ReleaseStatus(status)
    raise newException(Exception, msg)

# Automatic resource management with destroy hooks
proc `=destroy`*(info: OrtMemoryInfoObj) =
  ## Automatically release memory info when it goes out of scope
  let ptrInfo = cast[OrtMemoryInfo](addr info)
  if ptrInfo != nil:
    ort_ReleaseMemoryInfo(ptrInfo)

proc `=destroy`*(value: OrtValueObj) =
  ## Automatically release tensor value when it goes out of scope
  let ptrValue = cast[OrtValue](addr value)
  if ptrValue != nil:
    ort_ReleaseValue(ptrValue)

proc `=destroy`*(session: OrtSessionObj) =
  ## Automatically release session when it goes out of scope
  let ptrSession = cast[OrtSession](addr session)
  if ptrSession != nil:
    ort_ReleaseSession(ptrSession)

proc `=destroy`*(typeInfo: OrtTypeInfoObj) =
  ## Automatically release type info when it goes out of scope
  let ptrTypeInfo = cast[OrtTypeInfo](addr typeInfo)
  if ptrTypeInfo != nil:
    ort_ReleaseTypeInfo(ptrTypeInfo)

proc `=destroy`*(tensorInfo: OrtTensorTypeAndShapeInfoObj) =
  ## Automatically release tensor shape info when it goes out of scope
  let ptrTensorInfo = cast[OrtTensorTypeAndShapeInfo](addr tensorInfo)
  if ptrTensorInfo != nil:
    ort_ReleaseTensorTypeAndShapeInfo(ptrTensorInfo)

proc `=destroy`*(options: OrtSessionOptionsObj) =
  ## Automatically release session options when it goes out of scope
  let ptrOptions = cast[OrtSessionOptions](addr options)
  if ptrOptions != nil:
    ort_ReleaseSessionOptions(ptrOptions)

proc `=destroy`*(allocator: OrtAllocatorObj) =
  ## Automatically release allocator when it goes out of scope
  let ptrAllocator = cast[OrtAllocator](addr allocator)
  if ptrAllocator != nil:
    ort_ReleaseAllocator(ptrAllocator)


