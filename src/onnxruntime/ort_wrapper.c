/* ort_wrapper.c - ONNX Runtime C API wrapper */
#include <onnxruntime/onnxruntime_c_api.h>
#include <stdlib.h>

static const OrtApi* g_ort = NULL;

static int ort_init(void) {
    if (!g_ort) {
        const OrtApiBase* base = OrtGetApiBase();
        if (!base) return -1;
        g_ort = base->GetApi(ORT_API_VERSION);
    }
    return g_ort ? 0 : -1;
}

/* Status */
const char* ort_GetErrorMessage(OrtStatus* status) {
    if (ort_init() != 0) return NULL;
    return g_ort->GetErrorMessage(status);
}

void ort_ReleaseStatus(OrtStatus* status) {
    if (ort_init() != 0) return;
    g_ort->ReleaseStatus(status);
}

/* Environment */
OrtStatus* ort_CreateEnv(OrtLoggingLevel log_severity_level, const char* logid, OrtEnv** out) {
    if (ort_init() != 0) return NULL;
    return g_ort->CreateEnv(log_severity_level, logid, out);
}

void ort_ReleaseEnv(OrtEnv* env) {
    if (ort_init() != 0 || !env) return;
    g_ort->ReleaseEnv(env);
}

/* Session */
OrtStatus* ort_CreateSession(OrtEnv* env, const char* model_path, OrtSessionOptions* options, OrtSession** out) {
    if (ort_init() != 0) return NULL;
    return g_ort->CreateSession(env, model_path, options, out);
}

void ort_ReleaseSession(OrtSession* session) {
    if (ort_init() != 0 || !session) return;
    g_ort->ReleaseSession(session);
}

OrtStatus* ort_Run(OrtSession* session, OrtRunOptions* run_options,
                   const char* const* input_names, const OrtValue* const* inputs, size_t input_len,
                   const char* const* output_names, size_t output_names_len, OrtValue** outputs) {
    if (ort_init() != 0) return NULL;
    return g_ort->Run(session, run_options, input_names, inputs, input_len, 
                      output_names, output_names_len, outputs);
}

/* SessionOptions */
OrtStatus* ort_CreateSessionOptions(OrtSessionOptions** options) {
    if (ort_init() != 0) return NULL;
    return g_ort->CreateSessionOptions(options);
}

void ort_ReleaseSessionOptions(OrtSessionOptions* options) {
    if (ort_init() != 0 || !options) return;
    g_ort->ReleaseSessionOptions(options);
}

OrtStatus* ort_EnableCpuMemArena(OrtSessionOptions* options) {
    if (ort_init() != 0) return NULL;
    return g_ort->EnableCpuMemArena(options);
}

/* Note: CUDA provider API changed in newer versions
 * For simplicity, we return NULL to indicate CUDA is not available in this wrapper
 */
OrtStatus* ort_SessionOptionsAppendExecutionProvider_CUDA(OrtSessionOptions* options, int device_id) {
    (void)options;
    (void)device_id;
    return NULL;  /* CUDA not available in this build */
}

OrtStatus* ort_SessionOptionsAppendExecutionProvider_CoreML(OrtSessionOptions* options) {
    if (ort_init() != 0) return NULL;
    return g_ort->SessionOptionsAppendExecutionProvider(options, "CoreML", NULL, NULL, 0);
}

/* MemoryInfo */
OrtStatus* ort_CreateCpuMemoryInfo(OrtAllocatorType type, OrtMemType mem_type, OrtMemoryInfo** out) {
    if (ort_init() != 0) return NULL;
    return g_ort->CreateCpuMemoryInfo(type, mem_type, out);
}

void ort_ReleaseMemoryInfo(OrtMemoryInfo* info) {
    if (ort_init() != 0 || !info) return;
    g_ort->ReleaseMemoryInfo(info);
}

/* Allocator */
OrtStatus* ort_GetAllocatorWithDefaultOptions(OrtAllocator** out) {
    if (ort_init() != 0) return NULL;
    return g_ort->GetAllocatorWithDefaultOptions(out);
}

void ort_ReleaseAllocator(OrtAllocator* allocator) {
    if (ort_init() != 0 || !allocator) return;
    g_ort->ReleaseAllocator(allocator);
}

/* Tensor */
OrtStatus* ort_CreateTensorWithDataAsOrtValue(OrtMemoryInfo* info, void* p_data, size_t p_data_len,
                                               int64_t* shape, size_t shape_len,
                                               ONNXTensorElementDataType type, OrtValue** out) {
    if (ort_init() != 0) return NULL;
    return g_ort->CreateTensorWithDataAsOrtValue(info, p_data, p_data_len, shape, shape_len, type, out);
}

void ort_ReleaseValue(OrtValue* value) {
    if (ort_init() != 0 || !value) return;
    g_ort->ReleaseValue(value);
}

OrtStatus* ort_GetTensorMutableData(OrtValue* value, void** out) {
    if (ort_init() != 0) return NULL;
    return g_ort->GetTensorMutableData(value, out);
}

OrtStatus* ort_GetTensorTypeAndShape(OrtValue* value, OrtTensorTypeAndShapeInfo** out) {
    if (ort_init() != 0) return NULL;
    return g_ort->GetTensorTypeAndShape(value, out);
}

/* TypeInfo */
OrtStatus* ort_GetTypeInfo(OrtValue* value, OrtTypeInfo** out) {
    if (ort_init() != 0) return NULL;
    return g_ort->GetTypeInfo(value, out);
}

void ort_ReleaseTypeInfo(OrtTypeInfo* type_info) {
    if (ort_init() != 0 || !type_info) return;
    g_ort->ReleaseTypeInfo(type_info);
}

OrtStatus* ort_CastTypeInfoToTensorInfo(OrtTypeInfo* type_info, OrtTensorTypeAndShapeInfo** out) {
    if (ort_init() != 0) return NULL;
    return g_ort->CastTypeInfoToTensorInfo(type_info, out);
}

void ort_ReleaseTensorTypeAndShapeInfo(OrtTensorTypeAndShapeInfo* info) {
    if (ort_init() != 0 || !info) return;
    g_ort->ReleaseTensorTypeAndShapeInfo(info);
}

OrtStatus* ort_GetDimensionsCount(OrtTensorTypeAndShapeInfo* info, size_t* out) {
    if (ort_init() != 0) return NULL;
    return g_ort->GetDimensionsCount(info, out);
}

OrtStatus* ort_GetDimensions(OrtTensorTypeAndShapeInfo* info, int64_t* dim_values, size_t dim_values_length) {
    if (ort_init() != 0) return NULL;
    return g_ort->GetDimensions(info, dim_values, dim_values_length);
}

OrtStatus* ort_GetTensorShapeElementCount(OrtTensorTypeAndShapeInfo* info, size_t* out) {
    if (ort_init() != 0) return NULL;
    return g_ort->GetTensorShapeElementCount(info, out);
}

/* Session introspection */
OrtStatus* ort_SessionGetInputCount(OrtSession* session, size_t* out) {
    if (ort_init() != 0) return NULL;
    return g_ort->SessionGetInputCount(session, out);
}

OrtStatus* ort_SessionGetOutputCount(OrtSession* session, size_t* out) {
    if (ort_init() != 0) return NULL;
    return g_ort->SessionGetOutputCount(session, out);
}

OrtStatus* ort_SessionGetInputName(OrtSession* session, size_t index, OrtAllocator* allocator, char** out) {
    if (ort_init() != 0) return NULL;
    return g_ort->SessionGetInputName(session, index, allocator, out);
}

OrtStatus* ort_SessionGetOutputName(OrtSession* session, size_t index, OrtAllocator* allocator, char** out) {
    if (ort_init() != 0) return NULL;
    return g_ort->SessionGetOutputName(session, index, allocator, out);
}

OrtStatus* ort_SessionGetInputTypeInfo(OrtSession* session, size_t index, OrtTypeInfo** out) {
    if (ort_init() != 0) return NULL;
    return g_ort->SessionGetInputTypeInfo(session, index, out);
}

OrtStatus* ort_SessionGetOutputTypeInfo(OrtSession* session, size_t index, OrtTypeInfo** out) {
    if (ort_init() != 0) return NULL;
    return g_ort->SessionGetOutputTypeInfo(session, index, out);
}
