## ort_bindings.nim
## Low-level Nim bindings for ONNX Runtime C API
##
## Path configuration:
##   -d:ortPath=PATH      - Set ONNX Runtime installation path (auto-adds include/ and lib/)
##
## Examples:
##   nim c your_app.nim                              # Use system default paths
##   nim c -d:ortPath=/opt/onnxruntime your_app.nim  # Use custom installation path

import std/os

const
  OrtApiVersion* {.intdefine.} = 24

# Pass API version to C wrapper
{.passC: "-DORT_API_VERSION=" & $OrtApiVersion.}

const
  # Platform-specific library names
  OrtLibName* = when defined(macosx): "libonnxruntime.dylib"
                elif defined(windows): "onnxruntime.dll"
                else: "libonnxruntime.so"

  # Optional ONNX Runtime installation path
  OrtPath {.strdefine.} = ""


when OrtPath != "":
  const base = OrtPath
  const inc = base / "include"
  const lib = base / "lib"

  {.passC: "-I" & inc.}
  {.passL: "-L" & lib.}

  when defined(windows):
    {.passL: "onnxruntime.lib".}
  else:
    {.passL: "-lonnxruntime".}

else:
  when defined(macosx):
    const prefix = gorge("brew --prefix onnxruntime")
    const inc = prefix / "include"
    const lib = prefix / "lib"

    {.passC: "-I" & inc.}
    {.passL: "-L" & lib.}
    {.passL: "-lonnxruntime".}
    {.passL: "-Wl,-rpath," & lib.}

  elif defined(linux):
    {.passL: "-lonnxruntime".}

  elif defined(windows):
    {.passL: "onnxruntime.lib".}

{.compile: "ort_wrapper.c".}


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

  # Opaque handle types
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


proc GetErrorMessage*(status: OrtStatusPtr): cstring {.importc: "ort_GetErrorMessage", cdecl.}
proc ReleaseStatus*(status: OrtStatusPtr) {.importc: "ort_ReleaseStatus", cdecl.}

proc CreateEnv*(log_severity_level: OrtLoggingLevel, logid: cstring, outs: ptr OrtEnv): OrtStatusPtr {.importc: "ort_CreateEnv", cdecl.}
proc ReleaseEnv*(env: OrtEnv) {.importc: "ort_ReleaseEnv", cdecl.}

proc CreateSession*(env: OrtEnv, model_path: cstring, options: OrtSessionOptions, outs: ptr OrtSession): OrtStatusPtr {.importc: "ort_CreateSession", cdecl.}
proc ReleaseSession*(session: OrtSession) {.importc: "ort_ReleaseSession", cdecl.}
proc Run*(session: OrtSession, run_options: OrtRunOptions, input_names: ptr cstring, inputs: ptr OrtValue, input_len: csize_t, output_names: ptr cstring, output_names_len: csize_t, outputs: ptr OrtValue): OrtStatusPtr {.importc: "ort_Run", cdecl.}

proc CreateSessionOptions*(options: ptr OrtSessionOptions): OrtStatusPtr {.importc: "ort_CreateSessionOptions", cdecl.}
proc ReleaseSessionOptions*(options: OrtSessionOptions) {.importc: "ort_ReleaseSessionOptions", cdecl.}
proc EnableCpuMemArena*(options: OrtSessionOptions): OrtStatusPtr {.importc: "ort_EnableCpuMemArena", cdecl.}
proc SessionOptionsAppendExecutionProvider_CUDA*(options: OrtSessionOptions, device_id: cint): OrtStatusPtr {.importc: "ort_SessionOptionsAppendExecutionProvider_CUDA", cdecl.}
proc SessionOptionsAppendExecutionProvider_CoreML*(options: OrtSessionOptions): OrtStatusPtr {.importc: "ort_SessionOptionsAppendExecutionProvider_CoreML", cdecl.}

proc CreateCpuMemoryInfo*(`type`: OrtAllocatorType, mem_type: OrtMemType, outs: ptr OrtMemoryInfo): OrtStatusPtr {.importc: "ort_CreateCpuMemoryInfo", cdecl.}
proc ReleaseMemoryInfo*(info: OrtMemoryInfo) {.importc: "ort_ReleaseMemoryInfo", cdecl.}

proc GetAllocatorWithDefaultOptions*(outs: ptr OrtAllocator): OrtStatusPtr {.importc: "ort_GetAllocatorWithDefaultOptions", cdecl.}
proc ReleaseAllocator*(allocator: OrtAllocator) {.importc: "ort_ReleaseAllocator", cdecl.}

proc CreateTensorWithDataAsOrtValue*(info: OrtMemoryInfo, p_data: pointer, p_data_len: csize_t, shape: ptr int64, shape_len: csize_t, `type`: ONNXTensorElementDataType, outs: ptr OrtValue): OrtStatusPtr {.importc: "ort_CreateTensorWithDataAsOrtValue", cdecl.}
proc ReleaseValue*(value: OrtValue) {.importc: "ort_ReleaseValue", cdecl.}
proc GetTensorMutableData*(value: OrtValue, outs: ptr pointer): OrtStatusPtr {.importc: "ort_GetTensorMutableData", cdecl.}
proc GetTensorTypeAndShape*(value: OrtValue, outs: ptr OrtTensorTypeAndShapeInfo): OrtStatusPtr {.importc: "ort_GetTensorTypeAndShape", cdecl.}

proc GetTypeInfo*(value: OrtValue, outs: ptr OrtTypeInfo): OrtStatusPtr {.importc: "ort_GetTypeInfo", cdecl.}
proc ReleaseTypeInfo*(type_info: OrtTypeInfo) {.importc: "ort_ReleaseTypeInfo", cdecl.}
proc CastTypeInfoToTensorInfo*(type_info: OrtTypeInfo, outs: ptr OrtTensorTypeAndShapeInfo): OrtStatusPtr {.importc: "ort_CastTypeInfoToTensorInfo", cdecl.}
proc ReleaseTensorTypeAndShapeInfo*(info: OrtTensorTypeAndShapeInfo) {.importc: "ort_ReleaseTensorTypeAndShapeInfo", cdecl.}
proc GetDimensionsCount*(info: OrtTensorTypeAndShapeInfo, outs: ptr csize_t): OrtStatusPtr {.importc: "ort_GetDimensionsCount", cdecl.}
proc GetDimensions*(info: OrtTensorTypeAndShapeInfo, dim_values: ptr int64, dim_values_length: csize_t): OrtStatusPtr {.importc: "ort_GetDimensions", cdecl.}
proc GetTensorShapeElementCount*(info: OrtTensorTypeAndShapeInfo, outs: ptr csize_t): OrtStatusPtr {.importc: "ort_GetTensorShapeElementCount", cdecl.}

proc SessionGetInputCount*(session: OrtSession, outs: ptr csize_t): OrtStatusPtr {.importc: "ort_SessionGetInputCount", cdecl.}
proc SessionGetOutputCount*(session: OrtSession, outs: ptr csize_t): OrtStatusPtr {.importc: "ort_SessionGetOutputCount", cdecl.}
proc SessionGetInputName*(session: OrtSession, index: csize_t, allocator: OrtAllocator, outs: ptr cstring): OrtStatusPtr {.importc: "ort_SessionGetInputName", cdecl.}
proc SessionGetOutputName*(session: OrtSession, index: csize_t, allocator: OrtAllocator, outs: ptr cstring): OrtStatusPtr {.importc: "ort_SessionGetOutputName", cdecl.}
proc SessionGetInputTypeInfo*(session: OrtSession, index: csize_t, outs: ptr OrtTypeInfo): OrtStatusPtr {.importc: "ort_SessionGetInputTypeInfo", cdecl.}
proc SessionGetOutputTypeInfo*(session: OrtSession, index: csize_t, outs: ptr OrtTypeInfo): OrtStatusPtr {.importc: "ort_SessionGetOutputTypeInfo", cdecl.}


proc checkStatus*(status: OrtStatusPtr) =
  ## Check status and raise exception on error
  if status != nil:
    let msg = $GetErrorMessage(status)
    ReleaseStatus(status)
    raise newException(CatchableError, msg)

proc `=destroy`*(info: OrtMemoryInfoObj) =
  let ptrInfo = cast[OrtMemoryInfo](addr info)
  if ptrInfo != nil:
    ReleaseMemoryInfo(ptrInfo)

proc `=destroy`*(value: OrtValueObj) =
  let ptrValue = cast[OrtValue](addr value)
  if ptrValue != nil:
    ReleaseValue(ptrValue)

proc `=destroy`*(session: OrtSessionObj) =
  let ptrSession = cast[OrtSession](addr session)
  if ptrSession != nil:
    ReleaseSession(ptrSession)

proc `=destroy`*(typeInfo: OrtTypeInfoObj) =
  let ptrTypeInfo = cast[OrtTypeInfo](addr typeInfo)
  if ptrTypeInfo != nil:
    ReleaseTypeInfo(ptrTypeInfo)

proc `=destroy`*(tensorInfo: OrtTensorTypeAndShapeInfoObj) =
  let ptrTensorInfo = cast[OrtTensorTypeAndShapeInfo](addr tensorInfo)
  if ptrTensorInfo != nil:
    ReleaseTensorTypeAndShapeInfo(ptrTensorInfo)

proc `=destroy`*(options: OrtSessionOptionsObj) =
  let ptrOptions = cast[OrtSessionOptions](addr options)
  if ptrOptions != nil:
    ReleaseSessionOptions(ptrOptions)

proc `=destroy`*(allocator: OrtAllocatorObj) =
  let ptrAllocator = cast[OrtAllocator](addr allocator)
  if ptrAllocator != nil:
    ReleaseAllocator(ptrAllocator)
