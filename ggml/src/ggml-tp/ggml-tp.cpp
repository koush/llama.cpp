#include "ggml-tp.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"
#include "ggml-cpp.h"
#include "ggml-cpu.h"
#include "ggml-tp-threadpool.h"
#include <cuda_runtime.h>

#include <cinttypes>
#include <string>
#include <map>
#include <cstring>
#include <vector>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <numeric>
#include <set>
#include <thread>
#include <functional>

#define NUM_DEVICES 1
// #define GGML_BACKEND_TP_VALIDATE 1

static std::vector<ggml_backend_dev_t> ggml_parallel_devices;
static std::vector<ggml_backend_t> ggml_parallel_backends;
static std::vector<std::thread> ggml_parallel_workers;
static struct ggml_backend_tp_threadpool ggml_device_threadpool;

// TP data structures

static ggml_guid_t ggml_backend_tp_guid() {
    static ggml_guid guid = {0xa4, 0x1f, 0x3c, 0xd9, 0x87, 0x6b, 0x91, 0x22, 0xde, 0x33, 0xab, 0x7c, 0x58, 0x44, 0x9e, 0x01};
    return &guid;
}


struct ggml_backend_tp_context {
};

#define TP_MAX_DEVICES 2

enum ggml_tp_split_type {
    GGML_TP_SPLIT_NONE = 0,
    GGML_TP_SPLIT_ROWS = 1,
    GGML_TP_SPLIT_COLUMNS = 2,
    GGML_TP_SPLIT_REDUCE = 3,
    GGML_TP_SPLIT_DIM2 = 4,
    GGML_TP_SPLIT_VIEW = 5,
};

struct ggml_tensor_parallel_extra {
    bool initialized;

    // these are the tensors that are on the underlying GPUs, they may or may not be split.
    // Because llama cpp does not provide the full graph up front, these tensors are
    // fully allocated on each backend device.
    // however, trying to map addresses to full vs split tensors to new address spaces
    // would be a significant endeavour, and may require changes to llama cpp graph building.
    // the savings were estimated as 110MB on a 72B model.
    ggml_tensor * tensors[TP_MAX_DEVICES];

    bool rejoined[TP_MAX_DEVICES];
    bool computed[TP_MAX_DEVICES];

    // flag for whether the tensors are split. split tensors will have a full tensor in converted_tensors.
    ggml_tp_split_type split_tensors;

    // this tensor needs to be rejoined for another op.
    bool has_rejoin;

    // this tensor does not support split ops and has a src that needs to be rejoined.
    bool needs_src_rejoin;

    size_t rejoined_buft_offsets[TP_MAX_DEVICES];

    // holds either split tensor views or full rejoined tensors depending on the owning tensor.
    ggml_tensor * converted_tensors[TP_MAX_DEVICES];

    // when a tensor is rejoined, it gathers the data from all the devices.
    // each device will provide a view of their full data for other devices as copy destinations.
    // these are only write views and do not have an allocation of their own.
    ggml_tensor * rejoined_tensor_views[TP_MAX_DEVICES][TP_MAX_DEVICES];

    ggml_tensor * reduce_op_tensors[TP_MAX_DEVICES];
    ggml_tensor * reduce_split_views[TP_MAX_DEVICES];

    ~ggml_tensor_parallel_extra() {
        for (size_t i = 0; i < TP_MAX_DEVICES; i++) {
            auto tensor = tensors[i];
            if (!tensor) {
                break;
            }
            delete tensor;
            tensors[i] = nullptr;
    }

        for (size_t i = 0; i < TP_MAX_DEVICES; i++) {
            auto rejoined_tensor = converted_tensors[i];
            if (!rejoined_tensor) {
                break;
            }
            delete rejoined_tensor;
            converted_tensors[i] = nullptr;
        }

        for (size_t i = 0; i < TP_MAX_DEVICES; i++) {
            for (int j = 0; j < TP_MAX_DEVICES; j++) {
                auto rejoined_tensor_view = rejoined_tensor_views[i][j];
                if (!rejoined_tensor_view) {
                    break;
                }
                delete rejoined_tensor_view;
                rejoined_tensor_views[i][j] = nullptr;
            }
        }

        for (size_t i = 0; i < TP_MAX_DEVICES; i++) {
            auto reduce_op_tensor = reduce_op_tensors[i];
            if (!reduce_op_tensor) {
                break;
            }
            delete reduce_op_tensor;
            reduce_op_tensors[i] = nullptr;
        }

        for (size_t i = 0; i < TP_MAX_DEVICES; i++) {
            auto reduce_op_view = reduce_split_views[i];
            if (!reduce_op_view) {
                break;
            }
            delete reduce_op_view;
            reduce_split_views[i] = nullptr;
        }
    }
};


static const char * ggml_backend_tp_name(ggml_backend_t backend) {
    return "TP";
    GGML_UNUSED(backend);
}

static void ggml_backend_tp_free(ggml_backend_t backend) {
    ggml_backend_tp_context * tp_ctx = (ggml_backend_tp_context *)backend->context;
    delete tp_ctx;
    delete backend;
}

static void ggml_backend_tp_synchronize(ggml_backend_t backend) {
    // for (size_t j = 0; j < ggml_parallel_devices.size(); j++) {
    //     auto backend = ggml_parallel_backends[j];
    //     if (backend->iface.synchronize) {
    //         backend->iface.synchronize(backend);
    //     }
    //     backend->iface.synchronize(backend);
    // }

    GGML_UNUSED(backend);
}

static const char * ggml_backend_tp_split_buffer_type_name(ggml_backend_buffer_type_t buft) {
    return "TPSplit";
    GGML_UNUSED(buft);
}

static bool ggml_backend_buft_is_tp_split(ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name == ggml_backend_tp_split_buffer_type_name;
}

static const char * ggml_backend_tp_buffer_type_name(ggml_backend_buffer_type_t buft) {
    return "TP";
    GGML_UNUSED(buft);
}


static bool ggml_backend_buft_is_tp(ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name == ggml_backend_tp_buffer_type_name;
}

static bool ggml_backend_tp_is_split(ggml_tensor * tensor) {
    if (!tensor->buffer) {
        return false;
    }

    if (ggml_backend_buft_is_tp_split(tensor->buffer->buft)) {
        return true;
    }

    if (tensor->buffer->buft->iface.get_name != ggml_backend_tp_buffer_type_name) {
        return false;
    }

    auto extra = (ggml_tensor_parallel_extra *)tensor->extra;
    if (extra->has_rejoin) {
        return false;
    }
    return extra->split_tensors;
}

static bool is_split_compatible(ggml_tensor * tensor) {
    auto op = tensor->op;
    if (op == GGML_OP_MUL_MAT || op == GGML_OP_MUL_MAT_ID) {
        auto src1 = tensor->src[1];
        if (src1->buffer && ggml_backend_buft_is_tp_split(src1->buffer->buft)) {
            return false;
        }
        auto src0 = tensor->src[0];
        return ggml_backend_buft_is_tp_split(src0->buffer->buft);
    }

    switch (op) {
        case GGML_OP_UNARY:
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
        case GGML_OP_ADD:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
        case GGML_OP_RESHAPE:
        case GGML_OP_PERMUTE:
        case GGML_OP_ROPE:
        case GGML_OP_FLASH_ATTN_EXT:
        case GGML_OP_CPY:
        case GGML_OP_GET_ROWS:
            return true;
        default:
            return false;
    }
}

static size_t ggml_align_size(size_t size, size_t alignment) {
    return (size + alignment - 1) & ~(alignment - 1);
}

static ggml_tensor* ggml_backend_tp_clone_tensor(const ggml_tensor * tensor) {
    ggml_tensor * wrapped = new ggml_tensor();
    ggml_set_name(wrapped, tensor->name);
    wrapped->type = (ggml_type) tensor->type;
    wrapped->flags = tensor->flags;

    for (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        wrapped->ne[i] = tensor->ne[i];
    }

    for (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        wrapped->nb[i] = tensor->nb[i];
    }

    wrapped->op = (ggml_op) tensor->op;
    for (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / sizeof(int32_t); i++) {
        wrapped->op_params[i] = tensor->op_params[i];
    }
    return wrapped;
}

struct ggml_split {
    size_t split[TP_MAX_DEVICES];
};

static ggml_split get_dim_splits(size_t dim) {
    size_t split = dim / ggml_parallel_devices.size();
    ggml_split splits;
    for (size_t i = 0; i < ggml_parallel_devices.size() - 1; i++) {
        splits.split[i] = split;
    }
    splits.split[ggml_parallel_devices.size() - 1] = dim - split * (ggml_parallel_devices.size() - 1);
    return splits;
}

static ggml_split get_col_splits(const ggml_tensor * tensor) {
    return get_dim_splits(tensor->ne[0]);
}

static ggml_split get_row_splits(const ggml_tensor * tensor) {
    return get_dim_splits(tensor->ne[1]);
}

static size_t ggml_backend_tp_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    // try to use a native alignment but it must be multiple of all the device alignments
    size_t lcm = 1;
    for (size_t i = 0; i < ggml_parallel_devices.size(); i++) {
        auto dev = ggml_parallel_devices[i];
        auto buffer_type = dev->iface.get_buffer_type(dev);
        lcm = std::lcm(lcm, buffer_type->iface.get_alignment(buffer_type));
    }
    return lcm;
    GGML_UNUSED(buft);
}

static ggml_status ensure_dim2_split(const ggml_tensor *src) {
    auto src_extra = (ggml_tensor_parallel_extra *)src->extra;
    if (src_extra->split_tensors) {
        if (src_extra->split_tensors != GGML_TP_SPLIT_DIM2) {
            GGML_ABORT("Tensor %s is already split as %d, but requested to be split as dim2.\n", src->name, src_extra->split_tensors);
        }
        // this tensor is already split, so we don't need to do anything
        return GGML_STATUS_SUCCESS;
    }
    if (src_extra->converted_tensors[0]) {
        return GGML_STATUS_SUCCESS;
    }

    // no actual conversion needs to take place, the split tensors can be
    // created by using offsets within the original tensor.
    auto splits = get_dim_splits(src->ne[2]);
    size_t offset = 0;
    for (size_t j = 0; j < ggml_parallel_devices.size(); j++) {
        auto split = ggml_backend_tp_clone_tensor(src);
        split->op = GGML_OP_NONE;
        src_extra->converted_tensors[j] = split;

        split->buffer = src_extra->tensors[j]->buffer;
        split->data = (char *) src_extra->tensors[j]->data + offset;

        // note that only the dimension needs to be changed, retaining the stride allows
        // using the original tensor data for the row split.
        split->ne[2] = splits.split[j];

        offset += src->nb[1] / src->ne[2] * splits.split[j];
    }
    return GGML_STATUS_SUCCESS;
}

static ggml_status ensure_row_split(const ggml_tensor *src) {
    auto src_extra = (ggml_tensor_parallel_extra *)src->extra;
    if (src_extra->split_tensors) {
        if (src_extra->split_tensors != GGML_TP_SPLIT_ROWS) {
            GGML_ABORT("Tensor %s is already split as %d, but requested to be split as rows.\n", src->name, src_extra->split_tensors);
        }
        // this tensor is already split, so we don't need to do anything
        return GGML_STATUS_SUCCESS;
    }
    if (src_extra->converted_tensors[0]) {
        return GGML_STATUS_SUCCESS;
    }

    // no actual conversion needs to take place, the split tensors can be
    // created by using offsets within the original tensor.
    auto splits = get_row_splits(src);
    size_t offset = 0;
    for (size_t j = 0; j < ggml_parallel_devices.size(); j++) {
        auto split = ggml_backend_tp_clone_tensor(src);
        split->op = GGML_OP_NONE;
        src_extra->converted_tensors[j] = split;

        split->buffer = src_extra->tensors[j]->buffer;
        split->data = (char *) src_extra->tensors[j]->data + offset;

        // note that only the dimension needs to be changed, retaining the stride allows
        // using the original tensor data for the row split.
        split->ne[1] = splits.split[j];

        offset += src->nb[0] / src->ne[1] * splits.split[j];
    }
    return GGML_STATUS_SUCCESS;
}

static ggml_status ensure_column_split(const ggml_tensor *src) {
    auto src_extra = (ggml_tensor_parallel_extra *)src->extra;
    if (src_extra->split_tensors) {
        if (src_extra->split_tensors != GGML_TP_SPLIT_COLUMNS) {
            GGML_ABORT("Tensor %s is already split as %d, but requested to be split as columns.\n", src->name, src_extra->split_tensors);
        }
        // this tensor is already split, so we don't need to do anything
        return GGML_STATUS_SUCCESS;
    }
    if (src_extra->converted_tensors[0]) {
        return GGML_STATUS_SUCCESS;
    }

    // no actual conversion needs to take place, the split tensors can be
    // created by using offsets within the original tensor.

    // unlike the matmult weight tensors which are rejoined column wise, when
    // splitting tensors for unary or arithmetic operations, split them row wise.
    auto splits = get_col_splits(src);

    size_t offset = 0;
    for (size_t j = 0; j < ggml_parallel_devices.size(); j++) {
        auto split = ggml_backend_tp_clone_tensor(src);
        split->op = GGML_OP_NONE;
        src_extra->converted_tensors[j] = split;
        
        split->buffer = src_extra->tensors[j]->buffer;
        split->data = (char *) src_extra->tensors[j]->data + offset;

        // note that only the dimension needs to be changed, retaining the stride allows
        // using the original tensor data for the column split.
        split->ne[0] = splits.split[j];

        offset += src->nb[1] / src->ne[0] * splits.split[j];
    }

    return GGML_STATUS_SUCCESS;
}

struct ggml_backend_tp_buffer_context {
    // flag for if this is a buffer context for split weights
    bool split = false;
    void * base_ptr;

    // tensors with GGML_OP_NONE are never split and have full tensors on each device.
    // these tensors may be used for input and may have set_tensor called on them.
    size_t input_buffer_sizes[TP_MAX_DEVICES];
    ggml_backend_buffer_t input_backend_buffers[TP_MAX_DEVICES];

    // all other tensors are part of the pipeline and may be split or full.
    // this can not be determined until the graph is fully constructed.
    ggml_backend_buffer_t backend_buffers[TP_MAX_DEVICES];

    // todo: switch from extra pointer to actually allocating the extra and referencing within the vector.
    std::vector<ggml_tensor_parallel_extra *> extras;
    size_t rejoined_buft_sizes[TP_MAX_DEVICES];
    ggml_backend_buffer_t rejoined_bufts[TP_MAX_DEVICES];

    ~ggml_backend_tp_buffer_context() {
        reset();

        for (size_t i = 0; i < TP_MAX_DEVICES; i++) {
            auto backend_buffer = backend_buffers[i];
            if (backend_buffer) {
                backend_buffer->iface.free_buffer(backend_buffer);
                backend_buffers[i] = nullptr;
            }

            backend_buffer = input_backend_buffers[i];
            if (backend_buffer) {
                backend_buffer->iface.free_buffer(backend_buffer);
                input_backend_buffers[i] = nullptr;
            }

            auto rejoined_buft = rejoined_bufts[i];
            if (rejoined_buft) {
                rejoined_buft->iface.free_buffer(rejoined_buft);
                rejoined_bufts[i] = nullptr;
            }
        }
    }

    void reset() {
        for (size_t i = 0; i < TP_MAX_DEVICES; i++) {
            auto backend_buffer = backend_buffers[i];
            if (backend_buffer && backend_buffer->iface.reset) {
                backend_buffer->iface.reset(backend_buffer);
            }

            backend_buffer = input_backend_buffers[i];
            if (backend_buffer && backend_buffer->iface.reset) {
                backend_buffer->iface.reset(backend_buffer);
            }

            rejoined_buft_sizes[i] = 0;
            input_buffer_sizes[i] = 0;
        }

        for (auto & extra : extras) {
            delete extra;
        }
        extras.clear();
    }

    ggml_status allocate_rejoin_buffers() {
        for (size_t i = 0; i < TP_MAX_DEVICES; i++) {
            auto rejoin_buft = rejoined_bufts[i];
            auto rejoin_buft_size = rejoined_buft_sizes[i];
            if (rejoin_buft && rejoin_buft->size < rejoin_buft_size) {
                rejoin_buft->iface.free_buffer(rejoin_buft);
                rejoin_buft = nullptr;
            }

            if (rejoin_buft_size && !rejoin_buft) {
                auto backend_buffer = backend_buffers[i];
                auto buffer_type = backend_buffer->buft;
                rejoin_buft = buffer_type->iface.alloc_buffer(buffer_type, rejoin_buft_size);
                if (!rejoin_buft) {
                    GGML_LOG_ERROR("Failed to allocate rejoin buffer %zu\n", i);
                    return GGML_STATUS_FAILED;
                }
                rejoined_bufts[i] = rejoin_buft;
            }
        }
        return GGML_STATUS_SUCCESS;
    }
};

static void ensure_reduce_split_views(const ggml_tensor *tensor) {
    ggml_tensor_parallel_extra *extra = (ggml_tensor_parallel_extra *)tensor->extra;
    if (extra->split_tensors != GGML_TP_SPLIT_REDUCE) {
        return;
    }
    if (extra->reduce_split_views[0]){ 
        return;
    }

    ggml_split splits = get_col_splits(tensor);
    size_t col_offset = 0;

    for (size_t j = 0; j < ggml_parallel_devices.size(); j++) {

        auto wrapped = extra->tensors[j];

        // create a col split view of the reduced tensor
        auto reduce_split_view = ggml_backend_tp_clone_tensor(wrapped);
        extra->reduce_split_views[j] = reduce_split_view;
        reduce_split_view->buffer = wrapped->buffer;
        reduce_split_view->view_src = wrapped;
        reduce_split_view->view_offs = col_offset * wrapped->nb[0];
        reduce_split_view->data = (char *)wrapped->data + reduce_split_view->view_offs;
        reduce_split_view->ne[0] = splits.split[j];

        col_offset += splits.split[j];
    }
}

static void ensure_rejoined(const ggml_tensor *reason, const ggml_tensor * src) {
    auto ctx = (ggml_backend_tp_buffer_context *)src->buffer->context;
    if (ctx->split) {
        GGML_ABORT("Tensor %s is a split buffer, but rejoin requested.\n", src->name);
    }
    auto src_extra = (ggml_tensor_parallel_extra *)src->extra;
    if (!src_extra->split_tensors) {
        // this tensor is not split, so we can't rejoin it
        return;
    }

    if (reason) {
        auto reason_extra = (ggml_tensor_parallel_extra *)reason->extra;
        if (!reason_extra->needs_src_rejoin) {
            reason_extra->needs_src_rejoin = true;
        }
    }

    if (src_extra->has_rejoin) {
        return;
    }
    src_extra->has_rejoin = true;

    // if (reason && reason != src) {
    //     printf("Rejoining tensor for %s %s\n", ggml_op_name(reason->op), ggml_op_name(src->op));
    // }
    // printf("rejoining tensor %s for %s %s\n", src->name, reason ? ggml_op_name(reason->op) : "none", ggml_op_name(src->op));

    const auto alignment = ggml_backend_tp_buffer_type_get_alignment(src->buffer->buft);

    auto reduce_scale = src_extra->split_tensors == GGML_TP_SPLIT_REDUCE ? ggml_parallel_devices.size() : 1;

    for (size_t j = 0; j < ggml_parallel_devices.size(); j++) {
        auto dev = ggml_parallel_devices[j];
        auto buffer_type = dev->iface.get_buffer_type(dev);

        auto tensor_size = buffer_type->iface.get_alloc_size(buffer_type, src);
        auto aligned_size = ggml_align_size(tensor_size, alignment) * reduce_scale;

        ggml_tensor * rejoined = ggml_backend_tp_clone_tensor(src);
        src_extra->converted_tensors[j] = rejoined;
        src_extra->rejoined_buft_offsets[j] = ctx->rejoined_buft_sizes[j];
        ctx->rejoined_buft_sizes[j] += aligned_size;


        // since this is an input, rewrite the op.
        rejoined->op = GGML_OP_NONE;
    }

    if (src_extra->split_tensors == GGML_TP_SPLIT_REDUCE) {
        for (size_t j = 0; j < ggml_parallel_devices.size(); j++) {
            auto dev = ggml_parallel_devices[j];
            auto buffer_type = dev->iface.get_buffer_type(dev);
            auto tensor_size = buffer_type->iface.get_alloc_size(buffer_type, src);
            auto aligned_size = ggml_align_size(tensor_size, alignment);

            auto rejoined = src_extra->converted_tensors[j];

            size_t reduce_offset = 0;
            for (size_t i = 0; i < ggml_parallel_devices.size(); i++) {
                auto view = ggml_backend_tp_clone_tensor(src);
                src_extra->rejoined_tensor_views[j][i] = view;

                view->op = GGML_OP_NONE;
                view->view_src = rejoined;
                // adjust the offset to the start of the last tensor
                view->view_offs = reduce_offset;

                reduce_offset += aligned_size;
            }
        }
    }
    else if (src_extra->split_tensors == GGML_TP_SPLIT_ROWS) {
        auto splits = get_row_splits(src);
        for (size_t j = 0; j < ggml_parallel_devices.size(); j++) {
            auto rejoined = src_extra->converted_tensors[j];

            size_t row_offset = 0;
            for (size_t i = 0; i < ggml_parallel_devices.size(); i++) {
                auto view = ggml_backend_tp_clone_tensor(src);
                src_extra->rejoined_tensor_views[j][i] = view;

                view->op = GGML_OP_NONE;
                view->view_src = rejoined;
                view->ne[1] = splits.split[i];
                // adjust the offset to the start of the row in the destination tensor
                view->view_offs = src->nb[1] * row_offset;

                row_offset += splits.split[j];
            }
        }
    }
    else if (src_extra->split_tensors == GGML_TP_SPLIT_COLUMNS) {
        // a typical tensor that is split across multiple devices is usually column split.
        // this is because the weight matrixes are transposed and row split, resulting in
        // column split resilt. this can not be concatenated memorywise (rowwise).
        // rather, the tensors must be copied back in columnwise.
        // a 4x4 tensor with 4 GPU holding 4x2 tensors each:
        // A A B B C C D D
        // A A B B C C D D
        // A A B B C C D D
        // A A B B C C D D
        auto splits = get_col_splits(src);
        for (size_t j = 0; j < ggml_parallel_devices.size(); j++) {
            auto rejoined = src_extra->converted_tensors[j];

            size_t col_offset = 0;
            for (size_t i = 0; i < ggml_parallel_devices.size(); i++) {
                auto view = ggml_backend_tp_clone_tensor(src);
                src_extra->rejoined_tensor_views[j][i] = view;

                view->op = GGML_OP_NONE;
                view->view_src = rejoined;
                view->ne[0] = splits.split[i];
                // adjust the offset to the start of the column in the destination tensor
                view->view_offs = src->nb[1] / src->ne[0] * col_offset;

                col_offset += splits.split[j];
            }
        }
    }
    else {
        GGML_ABORT("Tensor %s is split as %d, but rejoin requested.\n", src->name, src_extra->split_tensors);
    }
}

static int memdiff_index(const void *a, const void *b, size_t length) {
    for (size_t i = 0; i < length; ++i) {
        if (((const char*)a)[i] != ((const char*)b)[i]) {
            return (int)i;  // return index of first difference
        }
    }
    return -1;  // no differences found
}

template<typename p>
static void read_tensor(ggml_tensor *tensor, std::unique_ptr<p, decltype(&std::free)> & memory) {
    auto tensor_size = ggml_nbytes(tensor);
    auto buffer = tensor->buffer;

    memory.reset((p*) malloc(tensor_size));

    buffer->iface.get_tensor(buffer, tensor, memory.get(), 0, tensor_size);
}

static void rejoin_tensor_data(const ggml_tensor * tensor, ggml_tensor_parallel_extra * extra, char *data) {
    auto recombined_size = ggml_nbytes(tensor);

    if (!extra->split_tensors) {
        return;
    }

    auto vs = tensor;
    while (vs->view_src) {
        vs = vs->view_src;
    }
    auto ne1 = vs->ne[1];
    auto nb1 = vs->nb[1];
    auto split_nb1 = get_dim_splits(nb1);

    auto rejoined_buffer = data;

    // a tensor that is split across 4 devices can not be concatenated memorywise (rowwise).
    // rather, the tensors must be copied back in columnwise.
    // a 4x4 tensor with 4 GPU holding 4x1 tensors each:
    // A B C D
    // A B C D
    // A B C D
    // A B C D
    for (int64_t row = 0; row < ne1; row++) {
        size_t offset = 0;
        auto data_row_offset = rejoined_buffer + row * nb1;
        for (size_t j = 0; j < ggml_parallel_devices.size(); j++) {
            auto wrapped = extra->tensors[j];
            auto wrapped_nb1 = split_nb1.split[j];
            auto wrapped_row_offset = wrapped_nb1 * row;
            auto be = ggml_parallel_backends[j];
            if (be->iface.get_tensor_async) {
                be->iface.get_tensor_async(be, wrapped, data_row_offset + offset, wrapped_row_offset, wrapped_nb1);
            }
            else {
                auto buft = wrapped->buffer;
                buft->iface.get_tensor(buft, wrapped, data_row_offset + offset, wrapped_row_offset, wrapped_nb1);
            }
            offset += wrapped_nb1;
        }
    }

    if (data) {
        return;
    }

    for (size_t j = 0; j < ggml_parallel_devices.size(); j++) {
        auto r = extra->converted_tensors[j];
        if (!r->buffer) {
            // TODO: watch for this, right now its valid to call rejoin_buffer for an output with no intention of
            // writing back to the device.
            // but this may be a silent failure.
            continue;
        }
        auto be = ggml_parallel_backends[j];
        if (be->iface.set_tensor_async) {
            be->iface.set_tensor_async(be, r, rejoined_buffer, 0, recombined_size);
        }
        else {
            auto buft = r->buffer;
            buft->iface.set_tensor(buft, r, rejoined_buffer, 0, recombined_size);
        }
    }
}

static void rejoin_tensor(const ggml_tensor * tensor, ggml_tensor_parallel_extra * extra) {
    return rejoin_tensor_data(tensor, extra, nullptr);
}

typedef struct compute_thread {
    int device_index;
    struct ggml_cgraph * cgraph;
    int end;
    ggml_backend_tp_semaphore semaphore;
    std::vector<struct compute_thread *> * peers;
    std::mutex done;
} compute_thread;

static void release_peers(struct compute_thread * thread) {
    for (size_t i = 0; i < thread->peers->size(); i++) {
        auto t = thread->peers->at(i);
        ggml_backend_tp_semaphore_release(&t->semaphore, 1);
    }
}

static ggml_status reduce_gathered_tensors(ggml_cgraph * backend_graph, int device_index, const ggml_tensor * tensor) {
    auto extra = (ggml_tensor_parallel_extra *)tensor->extra;
    if (extra->split_tensors != GGML_TP_SPLIT_REDUCE) {
        return GGML_STATUS_SUCCESS;
    }

    if (!extra->has_rejoin) {
        return GGML_STATUS_SUCCESS;
    }

    ggml_tensor * wrapped = extra->tensors[device_index];

    // when reducing a tensor, the actual op (sub or add) is contained in reduce_op_tensors
    // which needs a split view of the reduce state sources.
    // and the final reduce (add) is contained in tensors.
    // todo: make this part of the graph.
    for (size_t i = 0; i < ggml_parallel_devices.size(); i++) {
        if (i == 0) {
            wrapped->src[0] = extra->rejoined_tensor_views[device_index][i++];
            wrapped->src[1] = extra->rejoined_tensor_views[device_index][i];
            wrapped->op = GGML_OP_ADD;
            backend_graph->nodes[backend_graph->n_nodes++] = wrapped;
            continue;
        }
        GGML_ABORT("TP: reduce gather tensor %s for device %d, but it has more than 2 sources.\n", tensor->name, device_index);
        wrapped->src[0] = wrapped;
        wrapped->src[1] = extra->rejoined_tensor_views[device_index][i];
        backend_graph->nodes[backend_graph->n_nodes++] = wrapped;
    }

    return GGML_STATUS_SUCCESS;
}

static void set_tensor(ggml_backend_t be, ggml_tensor * tensor, float value) {
    std::unique_ptr<float, decltype(&std::free)> data(static_cast<float*>(std::malloc(ggml_nbytes(tensor))), &std::free);

    for (int64_t i = 0; i < ggml_nelements(tensor); i++) {
        data.get()[i] = value;
    }
    be->iface.set_tensor_async(be, tensor, data.get(), 0, ggml_nbytes(tensor));
}

static ggml_tensor* ggml_backend_tp_node_compute_split(int device_index, ggml_tensor * tensor) {
    auto extra = (ggml_tensor_parallel_extra *)tensor->extra;

    auto wrapped = extra->tensors[device_index];
    if (tensor->op != GGML_OP_ADD || extra->split_tensors != GGML_TP_SPLIT_REDUCE) {
       return wrapped;
    }

    auto reduce_op_tensor = extra->reduce_op_tensors[device_index];
    return reduce_op_tensor;
}

static bool immediate_compute = false;
static void ggml_backend_tp_buffer_compute_graph(ggml_cgraph * cgraph, std::function<bool(int, std::set<ggml_tensor*>)> gather_pending, std::function<bool(int, ggml_tensor *, ggml_tensor_parallel_extra *)> compute, std::function<void(int, std::set<ggml_tensor*>)> flush_compute) {
    std::set<ggml_tensor*> pending_gathers;
    for (int node_index = 0; node_index < cgraph->n_nodes; node_index++) {
        auto tensor = cgraph->nodes[node_index];
        auto extra = (ggml_tensor_parallel_extra *)tensor->extra;

        // wait for async memcpy to finish if needed
        if (extra->needs_src_rejoin && pending_gathers.size()) {
            if (!immediate_compute) {
                if (flush_compute) {
                    flush_compute(node_index, pending_gathers);
                }
            }

            if (!gather_pending(node_index, pending_gathers)) {
                return;
            }
            pending_gathers.clear();
        }

        if (!compute(node_index, tensor, extra)) {
            return;
        }

        if (extra->has_rejoin && extra->split_tensors != GGML_TP_SPLIT_VIEW) {
            pending_gathers.insert(tensor);
        }

        if (immediate_compute) {
            if (flush_compute) {
                flush_compute(node_index, pending_gathers);
            }
        }
    }
}

static void ggml_backend_tp_buffer_graph_compute_one(struct compute_thread * thread) {
    auto startTime = std::chrono::high_resolution_clock::now();
    GGML_UNUSED(startTime);
    auto cgraph = thread->cgraph;

    struct ggml_init_params params = {
        /* .mem_size   = */ ggml_tensor_overhead()*(cgraph->n_nodes + cgraph->n_leafs)*4 + ggml_graph_overhead_custom(cgraph->size, false),
        /* .mem_buffer = */ NULL,
        /* .no_alloc   = */ true
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    auto ctx = ctx_ptr.get();
    struct ggml_cgraph * backend_graph = ggml_new_graph_custom(ctx, cgraph->size, false);
    auto device_index = thread->device_index;
    auto be = ggml_parallel_backends[device_index];

    if (!be->iface.cpy_tensor2d_async) {
        GGML_ABORT("Backend %s does not support async tensor copy.\n", be->iface.get_name(be));
    }

    int rejoins = 0;

    auto flush_compute = [&](int node_index, std::set<ggml_tensor*> pending_gathers) {
        if (backend_graph->n_nodes ) {
            auto status = be->iface.graph_compute(be, backend_graph);
            // if (device_index == 0) {
            //     for (int i = 0; i < backend_graph->n_nodes; i++) {
            //         auto tensor = backend_graph->nodes[i];
            //         printf("TP %d: %s %s %x\n", node_index - backend_graph->n_nodes + i, ggml_op_name(tensor->op), tensor->name, tensor->data);
            //         for (int j = 0; j < GGML_MAX_SRC; j++) {
            //             auto src = tensor->src[j];
            //             if (!src) {
            //                 break;
            //             }
            //             if (src) {
            //                 printf("  src %d: %s %s %x\n", j, ggml_op_name(src->op), src->name, src->data);
            //             }
            //         }
            //     }
            // }
            if (status != GGML_STATUS_SUCCESS) {
                GGML_ABORT("Backend %s failed to compute graph %d with status %d\n", be->iface.get_name(be), node_index, status);
            }
            backend_graph->n_nodes = 0;
        }
        thread->end = node_index;

        for (auto & tensor : pending_gathers) {
            auto extra = (ggml_tensor_parallel_extra *)tensor->extra;
            auto wrapped = extra->tensors[device_index];

            // async copies
            for (size_t other_device_index = 0; other_device_index < ggml_parallel_devices.size(); other_device_index++) {
                auto other_be = ggml_parallel_backends[other_device_index];
                auto rejoined_tensor_view = extra->rejoined_tensor_views[other_device_index][device_index];
                if (!rejoined_tensor_view) {
                    break;
                }

                auto view_src = wrapped;
                while (view_src->view_src) {
                    view_src = view_src->view_src;
                }
                if (!be->iface.cpy_tensor2d_async(be, other_be, view_src, rejoined_tensor_view)) {
                    GGML_ABORT("Failed to copy tensor %s from device %d to device %ld\n", tensor->name, device_index, other_device_index);
                    // TODO, this is recoverable if something like this is implemented:
                    // ggml_backend_tensor2d_copy(view_src, rejoined_tensor_view);
                }
            }
        }
    };

    auto gather_pending = [&](int node_index, std::set<ggml_tensor*> pending_gathers) {
        rejoins++;
        // synchronize self and then release peers
        ggml_backend_synchronize(be);
        release_peers(thread);

        // wait for everyone else
        for (size_t i = 0; i < thread->peers->size(); i++) {
            ggml_backend_tp_semaphore_acquire(&thread->semaphore);
        }

        // once all peers are done, we can rejoin the tensors
        for (auto & pending : pending_gathers) {
            reduce_gathered_tensors(backend_graph, device_index, pending);
            auto pending_extra = (ggml_tensor_parallel_extra *)pending->extra;
            pending_extra->rejoined[device_index] = true;
        }
        return true;
        GGML_UNUSED(node_index);
    };

    auto compute = [&](int node_index, ggml_tensor * tensor, ggml_tensor_parallel_extra * extra) {
        auto wrapped = ggml_backend_tp_node_compute_split(device_index, tensor);
        if (extra->split_tensors != GGML_TP_SPLIT_VIEW) {
            backend_graph->nodes[backend_graph->n_nodes++] = wrapped;
        }
        extra->computed[device_index] = true;

        return true;
        GGML_UNUSED(node_index);
    };

    ggml_backend_tp_buffer_compute_graph(cgraph, gather_pending, compute, flush_compute);
    flush_compute(cgraph->n_nodes, std::set<ggml_tensor*>());

    thread->done.unlock();

    // printf("TP graph %d compute time: %ld us %d rejoins\n", device_index, std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - startTime).count(), rejoins);
}

static enum ggml_status ggml_backend_tp_finish_init_tensor(ggml_tensor *tensor);
static void ensure_weight_column_split(ggml_tensor * weight);

static void do_init(size_t node_index, ggml_tensor * tensor, ggml_tensor_parallel_extra * extra) {
    if (extra->initialized) {
        return;
    }

    extra->initialized = true;

    // ensure all src are initialized, out of order usage is possible on view tensors.
    for (size_t i = 0; i < GGML_MAX_SRC; i++) {
        auto src = tensor->src[i];
        if (!src) {
            break;
        }
        auto src_extra = (ggml_tensor_parallel_extra *)src->extra;
        // node index is incorrect here, but need something for debugging.
        do_init(node_index, src, src_extra);
    }


    auto src0 = tensor->src[0];
    auto src1 = tensor->src[1];
    auto src2 = tensor->src[2];
    auto src3 = tensor->src[3];
    auto src0_extra = src0 ? (ggml_tensor_parallel_extra *)src0->extra : nullptr;
    auto src1_extra = src1 ? (ggml_tensor_parallel_extra *)src1->extra : nullptr;
    auto src2_extra = src2 ? (ggml_tensor_parallel_extra *)src2->extra : nullptr;
    auto src3_extra = src3 ? (ggml_tensor_parallel_extra *)src3->extra : nullptr;

    auto create_default_tensors_for = [](ggml_tensor * tensor, ggml_tensor_parallel_extra * extra) {
        extra->split_tensors = GGML_TP_SPLIT_NONE;
        for (size_t j = 0; j < ggml_parallel_devices.size(); j++) {
            auto wrapped = ggml_backend_tp_clone_tensor(tensor);
            extra->tensors[j] = wrapped;
        }
    };

    auto create_default_tensors = [&]() {
        create_default_tensors_for(tensor, extra);
    };

    auto create_reduce_tensors = [&]() {
        extra->split_tensors = GGML_TP_SPLIT_REDUCE;
        for (size_t j = 0; j < ggml_parallel_devices.size(); j++) {
            auto wrapped = ggml_backend_tp_clone_tensor(tensor);
            extra->tensors[j] = wrapped;
        }
    };

    auto prepare_wrapped = [](ggml_tensor * tensor, ggml_tensor * dims) {
        auto wrapped = ggml_backend_tp_clone_tensor(dims);
        if (dims != tensor) {
            wrapped->op = tensor->op;
            for (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / sizeof(int32_t); i++) {
                wrapped->op_params[i] = tensor->op_params[i];
            }
        }
        return wrapped;
    };

    auto create_row_split_tensors_for = [prepare_wrapped](ggml_tensor * tensor, ggml_tensor_parallel_extra * extra, ggml_tensor * dims = nullptr) {
        dims = dims ? dims : tensor;
        extra->split_tensors = GGML_TP_SPLIT_ROWS;
        auto splits = get_row_splits(dims);
        for (size_t j = 0; j < ggml_parallel_devices.size(); j++) {
            auto wrapped = prepare_wrapped(tensor, dims);
            extra->tensors[j] = wrapped;

            // update row count
            wrapped->ne[1] = splits.split[j];
            // adjust the stride for the new row count
            wrapped->nb[2] = wrapped->nb[2] / dims->ne[1] * splits.split[j];
            wrapped->nb[3] = wrapped->nb[3] / dims->ne[1] * splits.split[j];
        }
    };

    auto create_row_split_tensors = [&]() {
        create_row_split_tensors_for(tensor, extra);
    };

    auto create_column_split_tensors_for = [prepare_wrapped](ggml_tensor * tensor, ggml_tensor_parallel_extra * extra, ggml_tensor * dims = nullptr) {
        dims = dims ? dims : tensor;
        extra->split_tensors = GGML_TP_SPLIT_COLUMNS;
        auto splits = get_col_splits(dims);
        for (size_t j = 0; j < ggml_parallel_devices.size(); j++) {
            auto wrapped = prepare_wrapped(tensor, dims);
            extra->tensors[j] = wrapped;

            // update col count
            wrapped->ne[0] = splits.split[j];
            // adjust the stride for the new col count
            wrapped->nb[1] = wrapped->nb[1] / dims->ne[0] * splits.split[j];
            wrapped->nb[2] = wrapped->nb[2] / dims->ne[0] * splits.split[j];
            wrapped->nb[3] = wrapped->nb[3] / dims->ne[0] * splits.split[j];
        }
    };

    auto create_column_split_tensors = [&]() {
        create_column_split_tensors_for(tensor, extra);
    };

    auto create_dim2_split_tensors_for = [prepare_wrapped](ggml_tensor * tensor, ggml_tensor_parallel_extra * extra, ggml_tensor * dims = nullptr) {
        dims = dims ? dims : tensor;
        extra->split_tensors = GGML_TP_SPLIT_DIM2;
        auto splits = get_dim_splits(dims->ne[2]);
        for (size_t j = 0; j < ggml_parallel_devices.size(); j++) {
            auto wrapped = prepare_wrapped(tensor, dims);
            extra->tensors[j] = wrapped;

            // update dim2 count
            wrapped->ne[2] = splits.split[j];
            // adjust the stride for the new dim2 count
            wrapped->nb[3] = wrapped->nb[3] / dims->ne[2] * splits.split[j];
        }
    };

    bool has_init = false;
    auto create_reduce_op_tensors = [&] {
        if (has_init) {
            GGML_ABORT("Tensor %s has already been initialized, but is being initialized again.\n", tensor->name);
        }
        has_init = true;
        ggml_backend_tp_finish_init_tensor(tensor);

        // one of these must be a reduce tensor, the other may be a split tensor, unsplit tensor, or even another reduce tensor.
        auto reduce_tensor = src0_extra->split_tensors == GGML_TP_SPLIT_REDUCE ? src0 : src1;
        auto add_tensor = src0_extra->split_tensors == GGML_TP_SPLIT_REDUCE ? src1 : src0;
        auto add_extra = (ggml_tensor_parallel_extra *)add_tensor->extra;
        auto reduce_extra = (ggml_tensor_parallel_extra *)reduce_tensor->extra;

        if (add_extra->split_tensors == GGML_TP_SPLIT_REDUCE) {
            // double reduce add can simply be added without any views.
            for (size_t j = 0; j < ggml_parallel_devices.size(); j++) {
                auto wrapped = extra->tensors[j];
                auto reduce_op = ggml_backend_tp_clone_tensor(wrapped);
                extra->reduce_op_tensors[j] = reduce_op;
                reduce_op->buffer = wrapped->buffer;
                reduce_op->view_src = wrapped;
                reduce_op->view_offs = 0;
                reduce_op->data = wrapped->data;

                reduce_op->src[0] = reduce_extra->has_rejoin ? reduce_extra->rejoined_tensor_views[j][j] : reduce_extra->tensors[j];
                reduce_op->src[1] = add_extra->has_rejoin ? add_extra->rejoined_tensor_views[j][j] : add_extra->tensors[j];
            }
        }
        else {
            auto splits = get_col_splits(tensor);
            size_t col_offset = 0;

            for (size_t j = 0; j < ggml_parallel_devices.size(); j++) {
                auto wrapped = extra->tensors[j];

                // create a col split view of the destination that can be used for reduction
                auto reduce_op = ggml_backend_tp_clone_tensor(wrapped);
                extra->reduce_op_tensors[j] = reduce_op;
                reduce_op->buffer = wrapped->buffer;
                reduce_op->view_src = wrapped;
                reduce_op->view_offs = col_offset * wrapped->nb[0];
                reduce_op->data = (char *)wrapped->data + reduce_op->view_offs;
                reduce_op->ne[0] = splits.split[j];

                // create a col split view of the reduced tensor
                ensure_reduce_split_views(reduce_tensor);

                auto reduce_op_src_view = reduce_extra->reduce_split_views[j];
                reduce_op->src[0] = reduce_op_src_view;

                auto add = add_extra->split_tensors == GGML_TP_SPLIT_NONE ? add_extra->converted_tensors[j] : add_extra->tensors[j];
                reduce_op->src[1] = add;

                col_offset += splits.split[j];

                if (!extra->reduce_op_tensors[j]->src[0] || !extra->reduce_op_tensors[j]->src[1]) {
                    GGML_ABORT("ggml_backend_tp_buffer_init_tensor: reduce op tensor %s has no src\n", tensor->name);
                }
            }
        }

    };

    auto no_reduce = [&](ggml_tensor *src, ggml_tensor_parallel_extra *src_extra) {
        if (src_extra->split_tensors == GGML_TP_SPLIT_REDUCE) {
            ensure_rejoined(tensor, src);
        }
    };

    auto no_split_view = [](ggml_tensor *src, ggml_tensor_parallel_extra *src_extra) {
        if (src_extra->split_tensors == GGML_TP_SPLIT_VIEW) {
            GGML_ABORT("Tensor %s has view source tensors, which are not supported for tensor parallelism.\n", src->name);
        }
    };

    auto set_src_tensor_for = [](ggml_tensor * tensor, ggml_tensor_parallel_extra * extra, int src_index, ggml_tp_split_type split) {
        auto src_extra = (ggml_tensor_parallel_extra *)tensor->src[src_index]->extra;
        // if (src_extra->has_read) {
        //     GGML_LOG_WARN("Tensor %s has already been initialized, but is being initialized again.\n", tensor->name);
        // }

        // auto src_split_tensors = src_extra->has_rejoin ? GGML_TP_SPLIT_NONE : src_extra->split_tensors;
        // if (src_split_tensors != src_extra->split_tensors) {
        //     GGML_ABORT("Tensor %s has src%d split as %d but requested to be split as %d.\n", tensor->name, src_index, src_split_tensors, src_extra->split_tensors);
        // }

        for (size_t j = 0; j < ggml_parallel_devices.size(); j++) {
            auto wrapped = extra->tensors[j];

            if (split == GGML_TP_SPLIT_NONE) {
                if (src_extra->split_tensors == GGML_TP_SPLIT_REDUCE) {
                    wrapped->src[src_index] = src_extra->tensors[j];
                }
                else if (src_extra->split_tensors) {
                    wrapped->src[src_index] = src_extra->converted_tensors[j];
                }
                else {
                    wrapped->src[src_index] = src_extra->tensors[j];
                }
            }
            else if (split == GGML_TP_SPLIT_ROWS || split == GGML_TP_SPLIT_COLUMNS || split == GGML_TP_SPLIT_DIM2) {
                if (src_extra->split_tensors == GGML_TP_SPLIT_NONE) {
                    wrapped->src[src_index] = src_extra->converted_tensors[j];
                }
                else if (src_extra->split_tensors == split) {
                    wrapped->src[src_index] = src_extra->tensors[j];
                }
                else if (src_extra->split_tensors == GGML_TP_SPLIT_REDUCE) {
                    wrapped->src[src_index] = src_extra->reduce_split_views[j];
                }
                else {
                    GGML_ABORT("Tensor %s has unsupported op %s for tensor parallelism, src%d is split as %d but requested to be split as %d.\n", tensor->name, ggml_op_name(tensor->op), src_index, src_extra->split_tensors, split);
                }
            }
            else if (split == GGML_TP_SPLIT_REDUCE) {
                if (src_extra->split_tensors == GGML_TP_SPLIT_REDUCE) {
                    wrapped->src[src_index] = src_extra->tensors[j];
                }
                else {
                    GGML_ABORT("Tensor %s has unsupported op %s for tensor parallelism, src%d is split as %d but requested to be split as %d.\n", tensor->name, ggml_op_name(tensor->op), src_index, src_extra->split_tensors, split);
                }
            }
            else {
                GGML_ABORT("NYI\n");
            }
        }
    };

    auto set_src_tensor = [&](int src_index, ggml_tp_split_type split) {
        set_src_tensor_for(tensor, extra, src_index, split);
    };

    auto check_srcs = [&]() {
        if (src0 && !extra->tensors[0]->src[0]) {
            GGML_ABORT("Tensor %s failed to create src0 tensors for tensor parallelism.\n", tensor->name);
        }
        if (src1 && !extra->tensors[0]->src[1]) {
            GGML_ABORT("Tensor %s failed to create src1 tensors for tensor parallelism.\n", tensor->name);
        }
        if (src2 && !extra->tensors[0]->src[2]) {
            GGML_ABORT("Tensor %s failed to create src2 tensors for tensor parallelism.\n", tensor->name);
        }
        if (src3 && !extra->tensors[0]->src[3]) {
            GGML_ABORT("Tensor %s failed to create src3 tensors for tensor parallelism.\n", tensor->name);
        }
    };

    auto ensure_init_from_viewsrc = [create_default_tensors_for, create_column_split_tensors_for, create_row_split_tensors_for, create_dim2_split_tensors_for](ggml_tensor * tensor, ggml_tensor_parallel_extra *extra) {
        if (extra->split_tensors != GGML_TP_SPLIT_VIEW) {
            return;
        }
        auto view_src = tensor->view_src;
        if (!view_src) {
            return;
        }
        auto view_src_extra = (ggml_tensor_parallel_extra *)view_src->extra;
        if (view_src_extra->split_tensors == GGML_TP_SPLIT_COLUMNS) {
            create_column_split_tensors_for(tensor, extra, view_src);
        }
        else if (view_src_extra->split_tensors == GGML_TP_SPLIT_ROWS) {
            create_row_split_tensors_for(tensor, extra, view_src);
        }
        else if (view_src_extra->split_tensors == GGML_TP_SPLIT_DIM2) {
            create_dim2_split_tensors_for(tensor, extra, view_src);
        }
        else if (view_src_extra->split_tensors == GGML_TP_SPLIT_NONE) {
            create_default_tensors_for(tensor, extra);
        }
        else {
            GGML_ABORT("Tensor %s has unsupported op %s for tensor parallelism, view_src is split as %d.\n", tensor->name, ggml_op_name(tensor->op), view_src_extra->split_tensors);
        }

        ggml_backend_tp_finish_init_tensor(tensor);
    };

    bool force_rejoin = false;
    if (force_rejoin) {
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            auto src = tensor->src[i];
            if (!src) {
                break;
            }
            auto ctx = (ggml_backend_tp_buffer_context *)src->buffer->context;
            if (ctx->split) {
                // this tensor is not split, so we can not rejoin it
                continue;
            }
            auto src_extra = (ggml_tensor_parallel_extra *)src->extra;
            if (src_extra->split_tensors) {
                ensure_rejoined(tensor, src);
            }
        }
    }

    switch (tensor->op) {
        case GGML_OP_ROPE: {
            auto src0_split_tensors = src0_extra->has_rejoin ? GGML_TP_SPLIT_NONE : src0_extra->split_tensors;

            if (src0_extra->split_tensors == GGML_TP_SPLIT_VIEW) {
                // make this into columns and create views into it
                auto src0_viewsrc = src0->view_src;
                auto src0_viewsrc_extra = (ggml_tensor_parallel_extra *)src0_viewsrc->extra;

                if (src0_extra->tensors[0]) {
                    GGML_ABORT("Reshape Tensor %s has already been initialized, but is being initialized again.\n", tensor->name);
                }

                if (src0_viewsrc_extra->split_tensors != GGML_TP_SPLIT_COLUMNS) {
                    GGML_ABORT("Tensor %s has unsupported op %s for tensor parallelism, src0 is split but not as columns.\n", tensor->name, ggml_op_name(tensor->op));
                }
                
                if (src0_viewsrc->ne[0] % tensor->ne[0]) {
                    GGML_ABORT("Tensor %s has unsupported op %s for tensor parallelism, src0 is split as view but not evenly divisible by the rope head count.\n", tensor->name, ggml_op_name(tensor->op));
                }

                // rope input is initially on columns.
                // input to rope is split [8192,1,1,1], per gpu it is [4096,1,1,1]
                // the input is then reshaped [128,64,1,1] per gpu it is [128,32,1,1]
                // this effectively splits it on the num heads 64->32 heads.
                // the output from rope is [128,64,1,1] per gpu it is [128,32,1,1]
                // this means that the rope output is now split on rows.
                src0_split_tensors = GGML_TP_SPLIT_ROWS;
                create_row_split_tensors_for(src0, src0_extra);
                set_src_tensor_for(src0, src0_extra, 0, GGML_TP_SPLIT_COLUMNS);
                ggml_backend_tp_finish_init_tensor(src0);
            }

            if (!src0_split_tensors) {
                create_default_tensors();
                set_src_tensor(0, GGML_TP_SPLIT_NONE);
                set_src_tensor(1, GGML_TP_SPLIT_NONE);
                if (src2) {
                    set_src_tensor(2, GGML_TP_SPLIT_NONE);
                }
            }
            else if (src0_split_tensors == GGML_TP_SPLIT_ROWS) {
                create_row_split_tensors();
                set_src_tensor(0, GGML_TP_SPLIT_ROWS);
                set_src_tensor(1, GGML_TP_SPLIT_NONE);
            }
            else {
                GGML_ABORT("Tensor %s has unsupported op %s for tensor parallelism, src0 is split but not as rows.\n", tensor->name, ggml_op_name(tensor->op));
                // technically, this is not a problem, but it is not expected.
                // ensure_rejoined(tensor, src0);
            }

            break;
        }

        case GGML_OP_FLASH_ATTN_EXT: {
            no_split_view(src1, src1_extra);
            no_split_view(src2, src2_extra);
            no_split_view(src3, src3_extra);
            if (tensor->view_src) {
                GGML_ABORT("Tensor %s has view source tensors, which are not supported for tensor parallelism.\n", tensor->name);
            }

            auto src0_split_tensors = src0_extra->has_rejoin ? GGML_TP_SPLIT_NONE : src0_extra->split_tensors;
            auto src1_split_tensors = src1_extra->has_rejoin ? GGML_TP_SPLIT_NONE : src1_extra->split_tensors;
            auto src2_split_tensors = src2_extra->has_rejoin ? GGML_TP_SPLIT_NONE : src2_extra->split_tensors;

            if (src0_split_tensors == GGML_TP_SPLIT_VIEW) {
                // make this into columns and create views into it
                auto src0_viewsrc = src0->view_src;
                auto src0_viewsrc_extra = (ggml_tensor_parallel_extra *)src0_viewsrc->extra;

                if (src0_extra->tensors[0]) {
                    GGML_ABORT("Reshape Tensor %s has already been initialized, but is being initialized again.\n", tensor->name);
                }

                if (src0_viewsrc_extra->split_tensors != GGML_TP_SPLIT_ROWS) {
                    GGML_ABORT("Tensor %s has unsupported op %s for tensor parallelism, src0 is split but not as rows.\n", tensor->name, ggml_op_name(tensor->op));
                }

                if (tensor->ne[1] % src0_viewsrc->ne[1]) {
                    GGML_ABORT("Tensor %s has unsupported op %s for tensor parallelism, src0 is split as view but not evenly divisible by the rope head count.\n", tensor->name, ggml_op_name(tensor->op));
                }

                // similar to ROPE above, the input must be dim2 split and becomes row split.
                src0_split_tensors = GGML_TP_SPLIT_DIM2;
                create_dim2_split_tensors_for(src0, src0_extra);
                set_src_tensor_for(src0, src0_extra, 0, GGML_TP_SPLIT_ROWS);
                if (src0->op == GGML_OP_PERMUTE) {
                    auto splits = get_dim_splits(src0->nb[1]);
                    for (size_t j = 0; j < ggml_parallel_devices.size(); j++) {
                        auto wrapped = src0_extra->tensors[j];
                        wrapped->nb[1] = splits.split[j];
                    }
                }
                ggml_backend_tp_finish_init_tensor(src0);
            }

            if (!src0_split_tensors && !src1_split_tensors && !src2_split_tensors) {
                create_default_tensors();
                set_src_tensor(0, GGML_TP_SPLIT_NONE);
                set_src_tensor(1, GGML_TP_SPLIT_NONE);
                set_src_tensor(2, GGML_TP_SPLIT_NONE);
                set_src_tensor(3, GGML_TP_SPLIT_NONE);
            }
            else {
                ensure_dim2_split(src0);
                ensure_dim2_split(src1);
                ensure_dim2_split(src2);
                create_row_split_tensors();
                set_src_tensor(0, GGML_TP_SPLIT_DIM2);
                set_src_tensor(1, GGML_TP_SPLIT_DIM2);
                set_src_tensor(2, GGML_TP_SPLIT_DIM2);
                set_src_tensor(3, GGML_TP_SPLIT_NONE);
            }
            check_srcs();
            break;
        }

        case GGML_OP_RMS_NORM:
            // auto src0_viewsrc = src0->view_src;
            // auto src0_viewsrc_extra = (ggml_tensor_parallel_extra *)src0_viewsrc->extra;
            // no_split_view(src0, src0_extra);
            if (src0->view_src) {
                ensure_rejoined(tensor, src0->view_src);
                create_default_tensors_for(src0, src0_extra);
                set_src_tensor_for(src0, src0_extra, 0, GGML_TP_SPLIT_NONE);
            }

            ensure_rejoined(tensor, src0);
            create_default_tensors();
            set_src_tensor(0, GGML_TP_SPLIT_NONE);
            check_srcs();
            break;

        case GGML_OP_ADD: {
            no_split_view(src0, src0_extra);
            no_split_view(src1, src1_extra);
            if (tensor->view_src) {
                GGML_ABORT("Tensor %s has view source tensors, which are not supported for tensor parallelism.\n", tensor->name);
            }

            auto src0_split_tensors = src0_extra->has_rejoin ? GGML_TP_SPLIT_NONE : src0_extra->split_tensors;
            auto src1_split_tensors = src1_extra->has_rejoin ? GGML_TP_SPLIT_NONE : src1_extra->split_tensors;

            if (!src0_split_tensors && !src1_split_tensors) {
                // straight add
                ensure_rejoined(tensor, src0);
                ensure_rejoined(tensor, src1);
                create_default_tensors();
                set_src_tensor(0, GGML_TP_SPLIT_NONE);
                set_src_tensor(1, GGML_TP_SPLIT_NONE);
            }
            else if ((src0_split_tensors || src1_split_tensors) && (src1_split_tensors != GGML_TP_SPLIT_REDUCE && src0_split_tensors != GGML_TP_SPLIT_REDUCE)) {
                auto split = src0_split_tensors ? src0_split_tensors : src1_split_tensors;
                if (split == GGML_TP_SPLIT_ROWS) {
                    ensure_row_split(src0);
                    ensure_row_split(src1);
                    create_row_split_tensors();
                    set_src_tensor(0, GGML_TP_SPLIT_ROWS);
                    set_src_tensor(1, GGML_TP_SPLIT_ROWS);
                }
                else if (src0_split_tensors == GGML_TP_SPLIT_COLUMNS) {
                    ensure_column_split(src0);
                    if (src1_extra->split_tensors == GGML_TP_SPLIT_REDUCE) {
                        ensure_reduce_split_views(src1);
                    }
                    else {
                        ensure_column_split(src1);
                    }
                    create_column_split_tensors();
                    set_src_tensor(0, GGML_TP_SPLIT_COLUMNS);
                    set_src_tensor(1, GGML_TP_SPLIT_COLUMNS);
                }
                else if (src0_split_tensors == GGML_TP_SPLIT_DIM2) {
                    ensure_dim2_split(src0);
                    ensure_dim2_split(src1);
                    create_row_split_tensors();
                    set_src_tensor(0, GGML_TP_SPLIT_DIM2);
                    set_src_tensor(1, GGML_TP_SPLIT_DIM2);
                }
                else {
                    GGML_ABORT("Tensor %s has unsupported op %s for tensor parallelism, src0 is split as %d but src1 is split as %d.\n", tensor->name, ggml_op_name(tensor->op), src0_split_tensors, src1_split_tensors);
                }
            }
            else if (src0_extra->split_tensors == GGML_TP_SPLIT_REDUCE && src1_extra->split_tensors == GGML_TP_SPLIT_REDUCE) {
                create_reduce_tensors();
                create_reduce_op_tensors();
            }
            else if (src0_split_tensors == GGML_TP_SPLIT_REDUCE && !src1_split_tensors) {
                // src1 may be a reduce split and later gathered, so check that.
                if (src1_extra->split_tensors != GGML_TP_SPLIT_REDUCE) {
                    ensure_column_split(src1);
                }
                create_reduce_tensors();
                create_reduce_op_tensors();
            }
            else if (src1_split_tensors == GGML_TP_SPLIT_REDUCE && !src0_split_tensors) {
                // src0 may be a reduce split and later gathered, so check that.
                if (src0_extra->split_tensors != GGML_TP_SPLIT_REDUCE) {
                    ensure_column_split(src0);
                }
                create_reduce_tensors();
                create_reduce_op_tensors();
            }
            else if (src0_split_tensors == GGML_TP_SPLIT_COLUMNS && !src1_split_tensors) {
                ensure_column_split(src0);
                ensure_column_split(src1);
                create_column_split_tensors();
                set_src_tensor(0, GGML_TP_SPLIT_COLUMNS);
                set_src_tensor(1, GGML_TP_SPLIT_COLUMNS);
            }
            else {
                // mismatched, so join both
                ensure_rejoined(tensor, src0);
                ensure_rejoined(tensor, src1);
                create_default_tensors();
                set_src_tensor(0, GGML_TP_SPLIT_NONE);
                set_src_tensor(1, GGML_TP_SPLIT_NONE);
            }

            if (extra->split_tensors == GGML_TP_SPLIT_REDUCE) {
                GGML_ASSERT(ggml_are_same_shape(extra->reduce_op_tensors[0], extra->reduce_op_tensors[0]->src[0]) && "Tensor parallel tensors must have the same shape.");
                GGML_ASSERT(extra->reduce_op_tensors[0]->src[0]->ne[0] == extra->reduce_op_tensors[0]->src[0]->ne[0] && "Tensor parallel has incorrect broadcast dimension (ne0).");
                GGML_ASSERT(extra->reduce_op_tensors[0]->ne[0] == extra->reduce_op_tensors[0]->src[1]->ne[0] && "Tensor parallel has incorrect broadcast dimension (ne1).");
            }
            else {
                GGML_ASSERT(ggml_are_same_shape(extra->tensors[0], extra->tensors[0]->src[0]) && "Tensor parallel tensors must have the same shape.");
                GGML_ASSERT(extra->tensors[0]->src[0]->ne[0] == extra->tensors[0]->src[0]->ne[0] && "Tensor parallel has incorrect broadcast dimension (ne0).");
                GGML_ASSERT(extra->tensors[0]->ne[0] == extra->tensors[0]->src[1]->ne[0] && "Tensor parallel has incorrect broadcast dimension (ne1).");
            }
            break;
        }

        case GGML_OP_UNARY: {
            no_split_view(src0, src0_extra);
            if (tensor->view_src) {
                GGML_ABORT("Tensor %s has view source tensors, which are not supported for tensor parallelism.\n", tensor->name);
            }

            no_reduce(src0, src0_extra);
            auto src0_split_tensors = src0_extra->has_rejoin ? GGML_TP_SPLIT_NONE : src0_extra->split_tensors;

            if (src0_split_tensors == GGML_TP_SPLIT_ROWS) {
                create_row_split_tensors();
                set_src_tensor(0, GGML_TP_SPLIT_ROWS);
            }
            else if (src0_split_tensors == GGML_TP_SPLIT_COLUMNS) {
                create_column_split_tensors();
                set_src_tensor(0, GGML_TP_SPLIT_COLUMNS);
            }
            else if (!src0_split_tensors) {
                create_default_tensors();
                set_src_tensor(0, GGML_TP_SPLIT_NONE);
            }
            else {
                GGML_ABORT("Tensor %s has unsupported op %s for tensor parallelism, src0 is split but not as columns or rows.\n", tensor->name, ggml_op_name(tensor->op));
            }

            GGML_ASSERT(ggml_are_same_shape(extra->tensors[0], extra->tensors[0]->src[0]) && "Tensor parallel tensors must have the same shape.");
            break;
        }

        case GGML_OP_SUB:
        case GGML_OP_DIV:
        case GGML_OP_MUL: {
            no_split_view(src0, src0_extra);
            no_split_view(src1, src1_extra);
            if (tensor->view_src) {
                GGML_ABORT("Tensor %s has view source tensors, which are not supported for tensor parallelism.\n", tensor->name);
            }

            no_reduce(src0, src0_extra);
            no_reduce(src1, src1_extra);
            auto src0_split_tensors = src0_extra->has_rejoin ? GGML_TP_SPLIT_NONE : src0_extra->split_tensors;
            auto src1_split_tensors = src1_extra->has_rejoin ? GGML_TP_SPLIT_NONE : src1_extra->split_tensors;

            if (src0_split_tensors == src1_split_tensors) {
                // spltis match
                if (src0_split_tensors == GGML_TP_SPLIT_COLUMNS) {
                    ensure_column_split(src0);
                    ensure_column_split(src1);
                    create_column_split_tensors();
                    set_src_tensor(0, GGML_TP_SPLIT_COLUMNS);
                    set_src_tensor(1, GGML_TP_SPLIT_COLUMNS);
                }
                else if (src0_split_tensors == GGML_TP_SPLIT_ROWS) {
                    ensure_row_split(src0);
                    ensure_row_split(src1);
                    create_row_split_tensors();
                    set_src_tensor(0, GGML_TP_SPLIT_ROWS);
                    set_src_tensor(1, GGML_TP_SPLIT_ROWS);
                }
                else if (src0_split_tensors == GGML_TP_SPLIT_NONE) {
                    ensure_rejoined(tensor, src0);
                    ensure_rejoined(tensor, src1);
                    create_default_tensors();
                    set_src_tensor(0, GGML_TP_SPLIT_NONE);
                    set_src_tensor(1, GGML_TP_SPLIT_NONE);
                }
                else {
                    GGML_ABORT("Tensor %s has unsupported op %s for tensor parallelism, src0 and src1 are split but not as columns or rows.\n", tensor->name, ggml_op_name(tensor->op));
                }
            }
            else if (src0_split_tensors && src1_split_tensors && src0_split_tensors != src1_split_tensors) {
                // splits do not match, so gather both. potentially possible to just gather one? is there a benefit to which?
                ensure_rejoined(tensor, src0);
                ensure_rejoined(tensor, src1);
                create_default_tensors();
                set_src_tensor(0, GGML_TP_SPLIT_NONE);
                set_src_tensor(1, GGML_TP_SPLIT_NONE);
            }
            else {
                // one split, one not split
                auto split_tensors = src0_split_tensors ? src0_split_tensors : src1_split_tensors;
                if (split_tensors == GGML_TP_SPLIT_COLUMNS) {
                    ensure_column_split(src0);
                    ensure_column_split(src1);
                    create_column_split_tensors();
                    set_src_tensor(0, GGML_TP_SPLIT_COLUMNS);
                    set_src_tensor(1, GGML_TP_SPLIT_COLUMNS);
                    // ensure_rejoined(nullptr, tensor);
                }
                else if (split_tensors == GGML_TP_SPLIT_ROWS) {
                    ensure_row_split(src0);
                    ensure_row_split(src1);
                    create_row_split_tensors();
                    set_src_tensor(0, GGML_TP_SPLIT_ROWS);
                    set_src_tensor(1, GGML_TP_SPLIT_ROWS);
                }
                else {
                    GGML_ABORT("Tensor %s has unsupported op %s for tensor parallelism, src0 is split but src1 is not.\n", tensor->name, ggml_op_name(tensor->op));
                }
            }

            GGML_ASSERT(ggml_are_same_shape(extra->tensors[0], extra->tensors[0]->src[0]) && "Tensor parallel tensors must have the same shape.");
            GGML_ASSERT(extra->tensors[0]->ne[0] == extra->tensors[0]->src[0]->ne[0] && "Tensor parallel has incorrect broadcast dimension (ne1).");
            break;
        }

        case GGML_OP_MUL_MAT_ID:
        case GGML_OP_MUL_MAT: {
            no_split_view(src0, src0_extra);
            if (tensor->view_src) {
                GGML_ABORT("Tensor %s has view source tensors, which are not supported for tensor parallelism.\n", tensor->name);
            }

            auto src0_ctx = (ggml_backend_tp_buffer_context *)src0->buffer->context;
            no_reduce(src0, src0_extra);
            no_reduce(src1, src1_extra);
            auto src0_split_tensors = src0_extra->has_rejoin ? GGML_TP_SPLIT_NONE : src0_extra->split_tensors;
            auto src1_split_tensors = src1_extra->has_rejoin ? GGML_TP_SPLIT_NONE : src1_extra->split_tensors;


            if (src1_split_tensors == GGML_TP_SPLIT_VIEW) {
                // make this into columns and create views into it
                auto src1_viewsrc = src1->view_src;
                auto src1_viewsrc_extra = (ggml_tensor_parallel_extra *)src1_viewsrc->extra;

                if (src1_extra->tensors[0]) {
                    GGML_ABORT("Reshape Tensor %s has already been initialized, but is being initialized again.\n", tensor->name);
                }

                if (src1_viewsrc_extra->split_tensors != GGML_TP_SPLIT_ROWS) {
                    GGML_ABORT("Tensor %s has unsupported op %s for tensor parallelism, src1 is split but not as rows.\n", tensor->name, ggml_op_name(tensor->op));
                }

                if ((src1_viewsrc->ne[0] * src1_viewsrc->ne[1]) % tensor->ne[0]) {
                    GGML_ABORT("Tensor %s has unsupported op %s for tensor parallelism, src1 is split as view but not evenly divisible by the rope head count.\n", tensor->name, ggml_op_name(tensor->op));
                }

                // similar to ROPE above, the input must be row split and becomes column split.
                src1_split_tensors = GGML_TP_SPLIT_COLUMNS;
                create_column_split_tensors_for(src1, src1_extra);
                ggml_backend_tp_finish_init_tensor(src1);
            }

            if (!src0_split_tensors && !src1_split_tensors) {
                create_default_tensors();
                set_src_tensor(0, GGML_TP_SPLIT_NONE);
                set_src_tensor(1, GGML_TP_SPLIT_NONE);
            }
            else if (src0_split_tensors == GGML_TP_SPLIT_ROWS && !src1_split_tensors) {
                // the large weight tensor is typically row split and multiplied with a full tensor resulting in a column split.
                create_column_split_tensors();
                set_src_tensor(0, GGML_TP_SPLIT_ROWS);
                set_src_tensor(1, GGML_TP_SPLIT_NONE);
            }
            else if (src0_ctx->split && src1_split_tensors == GGML_TP_SPLIT_COLUMNS) {
                if (false ){
                    // not possible
                    auto src0_ctx = (ggml_backend_tp_buffer_context *)src0->buffer->context;
                    if (!src0_ctx->split) {
                        ensure_rejoined(tensor, src0);
                    }
                    ensure_rejoined(tensor, src1);
                    create_column_split_tensors();
                    set_src_tensor(0, GGML_TP_SPLIT_ROWS);
                    set_src_tensor(1, GGML_TP_SPLIT_NONE);
                }
                else {
                    // a weight matrix is multiplied by a column split tensor (prior to ROPE), it can be massaged to a column split.
                    // this results in a reduce split.
                    ensure_weight_column_split(src0);
                    create_reduce_tensors();
                    set_src_tensor(0, GGML_TP_SPLIT_COLUMNS);
                    set_src_tensor(1, GGML_TP_SPLIT_COLUMNS);
                }
            }
            else if (src0_split_tensors == GGML_TP_SPLIT_COLUMNS && src1_split_tensors == GGML_TP_SPLIT_COLUMNS) {
                // technically supported like the weights above, but not expected.
                GGML_ABORT("Tensor %s has unsupported op %s for tensor parallelism, src0 and src1 are both split as columns.\n", tensor->name, ggml_op_name(tensor->op));
                // column split and column split result in a reduce split.
                create_reduce_tensors();
                set_src_tensor(0, GGML_TP_SPLIT_COLUMNS);
                set_src_tensor(1, GGML_TP_SPLIT_COLUMNS);
            }
            else if (src0_split_tensors == GGML_TP_SPLIT_ROWS && src1_split_tensors == GGML_TP_SPLIT_ROWS) {
                // not possible
                ensure_rejoined(tensor, src0);
                ensure_rejoined(tensor, src1);
                create_default_tensors();
                set_src_tensor(0, GGML_TP_SPLIT_NONE);
                set_src_tensor(1, GGML_TP_SPLIT_NONE);
            }
            else if (src0_split_tensors) {
                GGML_ABORT("Tensor %s has unsupported op %s for tensor parallelism, src0 is split but src1 is not.\n", tensor->name, ggml_op_name(tensor->op));
            }
            else if (src1_split_tensors) {
                GGML_ABORT("Tensor %s has unsupported op %s for tensor parallelism, src1 is split but src0 is not.\n", tensor->name, ggml_op_name(tensor->op));
            }
            else {
                GGML_ABORT("Tensor %s has unsupported op %s for tensor parallelism.\n", tensor->name, ggml_op_name(tensor->op));
            }

            GGML_ASSERT(extra->tensors[0]->src[0]->ne[0] == extra->tensors[0]->src[0]->ne[0] && "Tensor parallel tensors must have the same inner dimension (ne0).");
            GGML_ASSERT(extra->tensors[0]->ne[0] == extra->tensors[0]->src[0]->ne[1] && "Tensor parallel has incorrect outer dimension (ne0).");

            if (tensor->op == GGML_OP_MUL_MAT_ID) {
                // all experts are split so all GPUs will run a portion of each expert.
                set_src_tensor(2, GGML_TP_SPLIT_NONE);

                GGML_ASSERT(extra->tensors[0]->ne[2] == extra->tensors[0]->src[1]->ne[2] && "Tensor parallel has incorrect outer dimension (ne1).");
            }
            else {
                GGML_ASSERT(extra->tensors[0]->ne[1] == extra->tensors[0]->src[1]->ne[1] && "Tensor parallel has incorrect outer dimension (ne1).");
            }
            break;
        }

        case GGML_OP_CPY: {
            // the src1 is the destination, and has already been created.
            // it maybe op NONE or op VIEW. without graph inspection.
            // it is possible to use this cpy op to make the src1 tensor tree
            // split, but this is simpler for now.
            // the min split amount in supports_opt also affects this.
            ensure_init_from_viewsrc(src0, src0_extra);
            ensure_init_from_viewsrc(src1, src1_extra);
            ensure_rejoined(tensor, src0);
            ensure_rejoined(tensor, src1);
            create_default_tensors();
            set_src_tensor(0, GGML_TP_SPLIT_NONE);
            set_src_tensor(1, GGML_TP_SPLIT_NONE);
            check_srcs();
            break;
        }

        case GGML_OP_GET_ROWS: {
            no_split_view(src0, src0_extra);
            no_split_view(src1, src1_extra);
            if (tensor->view_src) {
                GGML_ABORT("Tensor %s has view source tensors, which are not supported for tensor parallelism.\n", tensor->name);
            }

            auto src0_split_tensors = src0_extra->has_rejoin ? GGML_TP_SPLIT_NONE : src0_extra->split_tensors;
            if (src0_split_tensors == GGML_TP_SPLIT_ROWS) {
                GGML_ABORT("Tensor %s has unsupported op %s for tensor parallelism, src0 is split as rows.\n", tensor->name, ggml_op_name(tensor->op));
                // technically supported, but not expected.
            }
            else if (src0_split_tensors == GGML_TP_SPLIT_COLUMNS) {
                create_column_split_tensors();
                set_src_tensor(0, GGML_TP_SPLIT_COLUMNS);
            }
            else if (!src0_split_tensors) {
                create_default_tensors();
                set_src_tensor(0, GGML_TP_SPLIT_NONE);
            }
            else if (src0_split_tensors == GGML_TP_SPLIT_REDUCE) {
                // this actually works out just fine because the rows can be gotten then added together.
                create_reduce_tensors();
                set_src_tensor(0, GGML_TP_SPLIT_REDUCE);
            }
            else {
                GGML_ABORT("Tensor %s has unsupported op %s for tensor parallelism, src0 is split but not as columns or rows.\n", tensor->name, ggml_op_name(tensor->op));
            }
            set_src_tensor(1, GGML_TP_SPLIT_NONE);
            check_srcs();
            break;
        }

        case GGML_OP_SUM_ROWS: {
            no_split_view(src0, src0_extra);
            if (extra->split_tensors == GGML_TP_SPLIT_COLUMNS) {
                create_column_split_tensors();
                set_src_tensor(0, GGML_TP_SPLIT_COLUMNS);
            }
            else {
                ensure_rejoined(tensor, src0);
                create_default_tensors();
                set_src_tensor(0, GGML_TP_SPLIT_NONE);
            }
            check_srcs();
            break;
        }
        case GGML_OP_SOFT_MAX:
        case GGML_OP_ARGSORT: {
            no_split_view(src0, src0_extra);

            ensure_rejoined(tensor, src0);
            create_default_tensors();
            set_src_tensor(0, GGML_TP_SPLIT_NONE);
            check_srcs();
            break;
        }

        case GGML_OP_VIEW: {
            auto src0_split_tensors = src0_extra->has_rejoin ? GGML_TP_SPLIT_NONE : src0_extra->split_tensors;
            if (!src0_split_tensors) {
                create_default_tensors();
                set_src_tensor(0, GGML_TP_SPLIT_NONE);
            }
            else {
                if (src0_split_tensors == GGML_TP_SPLIT_COLUMNS && src0->ne[0] == tensor->ne[0]) {
                    GGML_LOG_WARN("UNUSED CODE PATH VIEW SPLIT COL\n");
                    // column split tensor with no change to columns
                    create_column_split_tensors();
                    set_src_tensor(0, GGML_TP_SPLIT_COLUMNS);
                }
                else if (src0_split_tensors == GGML_TP_SPLIT_ROWS && src0->ne[1] == tensor->ne[1]) {
                    GGML_LOG_WARN("UNUSED CODE PATH VIEW SPLIT ROW\n");
                    // row split tensor with no change to rows
                    create_row_split_tensors();
                    set_src_tensor(0, GGML_TP_SPLIT_ROWS);
                }
                else {
                    has_init = true;
                    extra->split_tensors = GGML_TP_SPLIT_VIEW;
                }
            }
            break;
        }
        case GGML_OP_PERMUTE:
        case GGML_OP_RESHAPE: {
            auto src0_split_tensors = src0_extra->has_rejoin ? GGML_TP_SPLIT_NONE : src0_extra->split_tensors;
            // if split, skip, make the downstream op make sense of it, as some graphs combine a bunch of reshapes/permutes/views.
            if (!src0_split_tensors) {
                create_default_tensors();
                set_src_tensor(0, GGML_TP_SPLIT_NONE);
            }
            else {
                // GGML_LOG_WARN("Tensor %s has unsupported op %s for tensor parallelism, src0 is split.\n", tensor->name, ggml_op_name(tensor->op));
                has_init = true;
                extra->split_tensors = GGML_TP_SPLIT_VIEW;
            }
            break;
        }
        default:
            GGML_ABORT("Tensor %s has unsupported op %s for tensor parallelism.\n", tensor->name, ggml_op_name(tensor->op));
    }

    if (!has_init) {
        ggml_backend_tp_finish_init_tensor(tensor);
    }
}

static enum ggml_status ggml_backend_tp_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    // static auto startTime = std::chrono::high_resolution_clock::now();
    // auto since_last = std::chrono::high_resolution_clock::now() - startTime;
    // startTime = std::chrono::high_resolution_clock::now();
    // printf("since last TP graph compute: %ld us\n", std::chrono::duration_cast<std::chrono::microseconds>(since_last).count());
    
    std::set<ggml_tensor*> tensors;

    ggml_backend_tp_buffer_compute_graph(cgraph, nullptr, [&](int node_index, ggml_tensor * tensor, ggml_tensor_parallel_extra * extra) {
        for (size_t j = 0; j < ggml_parallel_devices.size(); j++) {
            extra->computed[j] = false;
            extra->rejoined[j] = false;
        }
        do_init(node_index, tensor, extra);
        tensors.insert(tensor);
        return true;
    }, nullptr);


    std::set<ggml_backend_tp_buffer_context *> contexts;
    for (auto tensor : tensors) {
        auto extra = (ggml_tensor_parallel_extra *)tensor->extra;

        if (extra->has_rejoin) {
            auto ctx = (ggml_backend_tp_buffer_context *)tensor->buffer->context;
            if (contexts.find(ctx) == contexts.end()) {
                contexts.insert(ctx);
                ctx->allocate_rejoin_buffers();
            }

            for (size_t j = 0; j < ggml_parallel_devices.size(); j++) {
                auto rejoined = extra->converted_tensors[j];
                    
                auto buft = ctx->rejoined_bufts[j];
                rejoined->buffer = buft;
                rejoined->data = (char *)buft->iface.get_base(buft) + extra->rejoined_buft_offsets[j];

                auto result = buft->iface.init_tensor(buft, rejoined);
                if (result != GGML_STATUS_SUCCESS) {
                    return result;
                }

                for (size_t i = 0; i < TP_MAX_DEVICES; i++) {
                    auto rejoined_tensor_view = extra->rejoined_tensor_views[j][i];
                    if (!rejoined_tensor_view) {
                        break;
                    }
                    rejoined_tensor_view->buffer = buft;
                    rejoined_tensor_view->data = (char *) rejoined->data + rejoined_tensor_view->view_offs;
                    rejoined_tensor_view->view_src = rejoined;
                }
            }
        }

        if (tensor->view_src) {
            // rewrite the data and buffer
            for (size_t j = 0; j < ggml_parallel_devices.size(); j++) {
                // GGML_TP_SPLIT_VIEW have no source and are not ever computed.
                auto wrapped = extra->tensors[j];
                if (!wrapped || !wrapped->src[0]) {
                    continue;
                }

                wrapped->data = (char *)wrapped->src[0]->data + wrapped->view_offs;
                wrapped->buffer = wrapped->src[0]->buffer;
            }
        }
    }


    std::vector<struct compute_thread *> threads;
    for (size_t j = 0; j < ggml_parallel_devices.size(); j++) {
        auto thread = new compute_thread();
        thread->device_index = j;
        thread->cgraph = cgraph;
        thread->peers = &threads;
        thread->done.lock();
        threads.push_back(thread);
    }

    for (auto thread : threads) {
        ggml_backend_tp_threadpool_enqueue(&ggml_device_threadpool, (thread_task_func)ggml_backend_tp_buffer_graph_compute_one, thread);
    }

    bool failed = false;
    for (auto thread : threads) {
        thread->done.lock();
        if (thread->end != cgraph->n_nodes) {
            failed = true;
            break;
        }
    }


    // for (size_t j = 0; j < ggml_parallel_devices.size(); j++) {
    //     auto be = ggml_parallel_backends[j];
    //     ggml_backend_synchronize(be);
    // }

    // auto endTime = std::chrono::high_resolution_clock::now();
    // printf("TP graph compute took %ld us\n", std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count());

    return failed ? GGML_STATUS_FAILED : GGML_STATUS_SUCCESS;

    GGML_UNUSED(backend);
}

static void ggml_backend_set_tensor_async_common(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    if (tensor->op != GGML_OP_NONE) {
        GGML_ABORT("ggml_backend_tp_buffer_set_tensor: tensor %s has unexpected op %s\n", tensor->name, ggml_op_name(tensor->op));
    }

    ggml_backend_tp_buffer_context * ctx = (ggml_backend_tp_buffer_context *)buffer->context;
    ggml_tensor_parallel_extra * extra = (ggml_tensor_parallel_extra *)tensor->extra;

    if (!ctx->split) {
        if (extra->split_tensors) {
            GGML_ABORT("ggml_backend_tp_buffer_set_tensor: tensor %s is split, but the context is not split\n", tensor->name);
        }
        if (tensor->op != GGML_OP_NONE) {
            GGML_ABORT("ggml_backend_tp_buffer_set_tensor: tensor %s has unexpected op %s\n", tensor->name, ggml_op_name(tensor->op));
        }

        for (size_t j = 0; j < ggml_parallel_devices.size(); j++) {
            auto wrapped = extra->tensors[j];
            auto be = ggml_parallel_backends[j];
            if (be->iface.set_tensor_async) {
                be->iface.set_tensor_async(be, wrapped, data, offset, size);
            }
            else {
                auto backend_buffer = ctx->backend_buffers[j];
                backend_buffer->iface.set_tensor(backend_buffer, wrapped, data, offset, size);
            }
        }
        return;
    }

    // weight matrices used for mul mat are transposed, so split on row
    ggml_split splits = get_row_splits(tensor);
    size_t cur_row = 0;
    for (size_t j = 0; j < ggml_parallel_devices.size(); j++) {
        auto wrapped = extra->tensors[j];
        auto be = ggml_parallel_backends[j];

        // the split tensors should have the same alignment as the wrapping tensor, and thus the same stride.
        if (wrapped->nb[1] != tensor->nb[1]) {
            GGML_LOG_ERROR("ggml_backend_tp_buffer_set_tensor: wrapped->nb[1] %zu != tensor->nb[1] %zu\n", wrapped->nb[1], tensor->nb[1]);
            return;
        }

        auto split_offset = cur_row * tensor->nb[1];
        auto split_size = (size_t) splits.split[j] * tensor->nb[1];
        
        if (be->iface.set_tensor_async) {
            be->iface.set_tensor_async(be, wrapped, (const char *) data + split_offset, 0, split_size);
        }
        else {
            auto backend_buffer = ctx->backend_buffers[j];
            backend_buffer->iface.set_tensor(backend_buffer, wrapped, (const char *) data + split_offset, 0, split_size);
        }

        cur_row += splits.split[j];
    }
}


static void ggml_backend_tp_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    ggml_backend_set_tensor_async_common(buf, tensor, data, offset, size);

    GGML_UNUSED(backend);
}

static void ggml_backend_tp_get_tensor_async(ggml_backend_t backend, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_tp_buffer_type());

    ggml_tensor_parallel_extra * extra = (ggml_tensor_parallel_extra *)tensor->extra;

    ensure_rejoined(nullptr, tensor);
    if (extra->split_tensors) {
        rejoin_tensor_data(tensor, extra, (char * )data);
        return;
    }

    for (size_t j = 0; j < ggml_parallel_devices.size(); j++) {
        auto be = ggml_parallel_backends[j];
        if (be->iface.get_tensor_async) {
            be->iface.get_tensor_async(be, extra->tensors[j], data, offset, size);
            return;
        }
    }

    // this call should not fail, so call sync version on first device...
    auto r = extra->tensors[0];
    auto buft = r->buffer;
    buft->iface.get_tensor(buft, r, data, offset, size);

    GGML_UNUSED(backend);
}

static bool ggml_backend_tp_cpy_tensor_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, const ggml_tensor * src, ggml_tensor * dst) {
    if (dst->buffer->buft != ggml_backend_tp_buffer_type()) {
        return false;
    }
    auto dst_extra = (ggml_tensor_parallel_extra *)dst->extra;
    if (dst_extra->split_tensors || dst_extra->split_tensors == GGML_TP_SPLIT_REDUCE) {
        GGML_ABORT("Tensor %s is split, but set_tensor_async is called\n", dst->name);
    }

    if (ggml_backend_buffer_is_host(src->buffer)) {
        ggml_backend_tp_set_tensor_async(backend_dst, dst, src->data, 0, ggml_nbytes(src));
        return true;
    }

    if (src->buffer->buft != ggml_backend_tp_buffer_type()) {
        return false;
    }

    auto src_extra = (ggml_tensor_parallel_extra *)src->extra;
    if (src_extra->split_tensors == GGML_TP_SPLIT_REDUCE) {
        GGML_ABORT("Tensor %s is reduced, but cpy_tensor_async is called\n", src->name);
    }

    // async copies first
    for (size_t j = 0; j < ggml_parallel_devices.size(); j++) {
        auto be_src = ggml_parallel_backends[j];
        if (!be_src->iface.cpy_tensor_async) {
            continue;
        }

        auto wrapped_src = src_extra->tensors[j];
        auto wrapped_dst = dst_extra->tensors[j];
        if (wrapped_src->buffer == wrapped_dst->buffer) {
            continue;
        }

        auto be_dst = ggml_parallel_backends[j];

        if (!be_src->iface.cpy_tensor_async(be_src, be_dst, wrapped_src, wrapped_dst)) {
            ggml_backend_tensor_copy(wrapped_src, wrapped_dst);
        }
    }

    // sync copies
    for (size_t j = 0; j < ggml_parallel_devices.size(); j++) {
        auto be_src = ggml_parallel_backends[j];
        if (be_src->iface.cpy_tensor_async) {
            continue;
        }

        auto wrapped_src = src_extra->tensors[j];
        auto wrapped_dst = dst_extra->tensors[j];

        ggml_backend_tensor_copy(wrapped_src, wrapped_dst);
    }

    return true;

    GGML_UNUSED(backend_src);
}

static ggml_backend_i ggml_backend_tp_interface = {
    /* .get_name                = */ ggml_backend_tp_name,
    /* .free                    = */ ggml_backend_tp_free,
    /* .set_tensor_async        = */ ggml_backend_tp_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_tp_get_tensor_async,
    /* .cpy_tensor_async        = */ ggml_backend_tp_cpy_tensor_async,
    /* .synchronize             = */ ggml_backend_tp_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_tp_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
    /* .set_tensor2d_async      = */ NULL,
    /* .get_tensor2d_async      = */ NULL,
    /* .cpy_tensor2d_async      = */ NULL,
};

static ggml_backend_dev_t ggml_backend_tp_reg_get_device(ggml_backend_reg_t reg, size_t index);

ggml_backend_t ggml_backend_tp_init(const char * endpoint) {
    ggml_backend_tp_context * ctx = new ggml_backend_tp_context {
    };

    ggml_backend_t backend = new ggml_backend {
        /* .guid      = */ ggml_backend_tp_guid(),
        /* .interface = */ ggml_backend_tp_interface,
        /* .device    = */ ggml_backend_tp_reg_get_device(ggml_backend_tp_reg(), 0),
        /* .context   = */ ctx
    };

    return backend;

    GGML_UNUSED(endpoint);
}


// device interface

struct ggml_backend_tp_device_context {
    std::string endpoint;
    std::string name;
};

static const char * ggml_backend_tp_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_tp_device_context * ctx = (ggml_backend_tp_device_context *)dev->context;

    return ctx->name.c_str();
}

static const char * ggml_backend_tp_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_tp_device_context * ctx = (ggml_backend_tp_device_context *)dev->context;

    return ctx->name.c_str();
}

static void ggml_backend_tp_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    ggml_backend_tp_device_context * ctx = (ggml_backend_tp_device_context *)dev->context;

    *free = 84294967296;
    *total = 84294967296;

    GGML_UNUSED(dev);
    GGML_UNUSED(ctx);
}

static enum ggml_backend_dev_type ggml_backend_tp_device_get_type(ggml_backend_dev_t dev) {
    // TODO: obtain value from the server
    return GGML_BACKEND_DEVICE_TYPE_GPU;

    GGML_UNUSED(dev);
}

static void ggml_backend_tp_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_tp_device_get_name(dev);
    props->description = ggml_backend_tp_device_get_description(dev);
    props->type        = ggml_backend_tp_device_get_type(dev);
    ggml_backend_tp_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer           = */ true,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ false,
    };
}

struct ggml_backend_tp_buffer_type_context {
    bool split;
};


static void ggml_backend_tp_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_tp_buffer_context * ctx = (ggml_backend_tp_buffer_context *)buffer->context;
    delete ctx;
}

static void * ggml_backend_tp_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_tp_buffer_context * ctx = (ggml_backend_tp_buffer_context *)buffer->context;
    return ctx->base_ptr;
}

static size_t ggml_alloc_align_size(size_t size, size_t index) {
    auto dev = ggml_parallel_devices[index];
    auto buffer_type = dev->iface.get_buffer_type(dev);
    auto alignment = buffer_type->iface.get_alignment(buffer_type);
    return ggml_align_size(size, alignment);
}

static ggml_split get_alloc_splits(size_t size) {
    size_t split = size / ggml_parallel_devices.size();
    ggml_split splits;
    for (size_t i = 0; i < ggml_parallel_devices.size() - 1; i++) {
        splits.split[i] = ggml_alloc_align_size(split, i);
    }
    auto last = ggml_parallel_devices.size() - 1;
    splits.split[ggml_parallel_devices.size() - 1] = ggml_alloc_align_size(size - split * last, last);
    return splits;
}

static void ensure_weight_column_split(ggml_tensor * weight) {
    if (weight->op != GGML_OP_NONE) {
        return;
    }

    auto extra = (ggml_tensor_parallel_extra *)weight->extra;
    if (!extra->split_tensors) {
        return;
    }

    if (extra->split_tensors == GGML_TP_SPLIT_COLUMNS) {
        return;
    }
    extra->split_tensors = GGML_TP_SPLIT_COLUMNS;

    // the weight tensor is currently split by rows, reassemble it
    auto size = ggml_nbytes(weight);
    std::unique_ptr<char, decltype(&std::free)> data(
        static_cast<char*>(std::malloc(size)), &std::free);
    size_t offset = 0;
    for (size_t j = 0; j < ggml_parallel_devices.size(); j++) {
        auto wrapped = extra->tensors[j];
        auto buft = wrapped->buffer;
        auto wrapped_size = ggml_nbytes(wrapped);
        auto be = ggml_parallel_backends[j];
        if (be->iface.get_tensor_async) {
            be->iface.get_tensor_async(be, wrapped, data.get() + offset, 0, wrapped_size);
        }
        else {
            buft->iface.get_tensor(buft, wrapped, data.get() + offset, 0, wrapped_size);
        }
        offset += ggml_nbytes(wrapped);
    }

    for (size_t j = 0; j < ggml_parallel_devices.size(); j++) {
        auto be = ggml_parallel_backends[j];
        if (be->iface.get_tensor_async) {
            ggml_backend_synchronize(be);
        }
    }

    // now split on columns
    auto block_size = ggml_blck_size(weight->type);
    auto blocks_per_row = weight->ne[0] / block_size;
    auto elements_per_block = weight->ne[0] / blocks_per_row;
    auto splits = get_dim_splits(blocks_per_row);

    offset = 0;
    for (size_t j = 0; j < ggml_parallel_devices.size(); j++) {
        auto wrapped = extra->tensors[j];
        wrapped->ne[0] = splits.split[j] * elements_per_block;
        wrapped->ne[1] = weight->ne[1];
        wrapped->nb[1] = wrapped->nb[0] * (wrapped->ne[0] / ggml_blck_size(wrapped->type));
        wrapped->nb[2] = wrapped->nb[1] * wrapped->ne[1];
        wrapped->nb[3] = wrapped->nb[2] * wrapped->ne[2];

        auto be = ggml_parallel_backends[j];
        if (be->iface.set_tensor2d_async) {
            be->iface.set_tensor2d_async(be, wrapped, data.get() + offset, wrapped->nb[1], wrapped->ne[1], weight->nb[1]);
        }
        else if (be->iface.set_tensor_async) {
            for (int i = 0; i < weight->ne[1]; i++) {
                be->iface.set_tensor_async(be, wrapped, data.get() + i * weight->nb[1] + offset, i * wrapped->nb[1], wrapped->nb[1]);
            }
        }
        else {
            std::unique_ptr<char, decltype(&std::free)> slice(
                static_cast<char*>(std::malloc(wrapped->nb[3])), &std::free);

            // blit from the full memory to a split memory
            for (int i = 0; i < weight->ne[1]; i++) {
                memcpy(slice.get() + i * wrapped->nb[1], data.get() + i * weight->nb[1] + offset, wrapped->nb[1]);
            }

            // set the tensor from the split memory
            auto buft = wrapped->buffer;
            auto buft_size = ggml_nbytes(wrapped);
            if (be->iface.set_tensor_async) {
                be->iface.set_tensor_async(be, wrapped, slice.get(), 0, buft_size);
            }
            else {
                buft->iface.set_tensor(buft, wrapped, slice.get(), 0, buft_size);
            }
        }
        // track the column offset
        offset += wrapped->nb[1];
    }
}

static enum ggml_status ggml_backend_tp_finish_init_tensor(ggml_tensor *tensor) {
    ggml_backend_buffer_t buffer = tensor->buffer;
    ggml_backend_tp_buffer_context * ctx = (ggml_backend_tp_buffer_context *)buffer->context;
    auto extra = (ggml_tensor_parallel_extra *)tensor->extra;

    for (size_t j = 0; j < ggml_parallel_devices.size(); j++) {
        auto wrapped = extra->tensors[j];
        auto backend_buffer = ctx->backend_buffers[j];
        wrapped->buffer = backend_buffer;

        const auto alignment = ggml_backend_tp_buffer_type_get_alignment(buffer->buft);
        auto device_alignment = ggml_backend_tp_buffer_type_get_alignment(backend_buffer->buft);

        if (!tensor->view_src) {
            // the virtual address of the buffer starts at the alignment value, so subtract that.
            auto tensor_base = (uint64_t) tensor->data - alignment;
            // tensor data is expected to be aligned.
            if (tensor_base % alignment) {
                GGML_ABORT("ggml_backend_tp_buffer_init_tensor: tensor %s is not aligned to %zu\n", tensor->name, alignment);
            }
            auto tensor_blocks = tensor_base / alignment;

            if (tensor_blocks % ggml_parallel_devices.size()) {
                GGML_ABORT("ggml_backend_tp_buffer_init_tensor: tensor %s is not aligned to device count %zu\n", tensor->name, alignment);
            }

            auto base = (char *) backend_buffer->iface.get_base(backend_buffer);

            size_t device_blocks;
            if (ctx->split) {
                device_blocks = tensor_blocks / ggml_parallel_devices.size();
            }
            else {
                device_blocks = tensor_blocks;
            }

            auto device_base_offset = device_blocks * device_alignment;
            wrapped->data = base + device_base_offset;
        }
        else {
            if (ctx->split) {
                GGML_ABORT("ggml_backend_tp_buffer_init_tensor: split buffer type %s does not support views\n", buffer->buft->iface.get_name(buffer->buft));
            }
            auto view_src_extra = (ggml_tensor_parallel_extra *)tensor->view_src->extra;
            auto view_src = view_src_extra->tensors[j];
            auto rem = tensor->view_offs % alignment;
            auto view_offs = tensor->view_offs / alignment * device_alignment + rem;
            wrapped->data = (char *) view_src->data + view_offs;
            wrapped->view_src = view_src;
            wrapped->view_offs = view_offs;
            if (wrapped->view_src == NULL) {
                GGML_ABORT("ggml_backend_tp_buffer_init_tensor: view_src is NULL for tensor %s\n", tensor->name);
            }
        }

        auto result = backend_buffer->iface.init_tensor(backend_buffer, wrapped);
        if (result != GGML_STATUS_SUCCESS) {
            GGML_ABORT("ggml_backend_tp_buffer_init_tensor: init_tensor failed for tensor %s\n", tensor->name);
        }
    }

    return GGML_STATUS_SUCCESS;
}

static enum ggml_status ggml_backend_tp_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    ggml_backend_tp_buffer_context * ctx = (ggml_backend_tp_buffer_context *)buffer->context;

    ggml_tensor_parallel_extra * extra = new ggml_tensor_parallel_extra();
    ctx->extras.push_back(extra);
    tensor->extra = extra;

    if (tensor->view_src) {
        // if this is a view tensor, it will be initialized later when the src is initialized.
        // this is to avoid initializing the view tensor before the src is initialized.
        return GGML_STATUS_SUCCESS;
    }

    // tensors with ops will be initialized later when determining the graph state.
    // no-op tensors are always full tensors, and must be initialized immediately as they may be used as input or weights
    // and will have set_tensor called on them.
    if (!ctx->split && tensor->op != GGML_OP_NONE) {
        return GGML_STATUS_SUCCESS;
    }

    if (extra->initialized) {
        return GGML_STATUS_SUCCESS;
    }
    extra->initialized = true;

    GGML_ASSERT(!tensor->view_src);

    for (size_t j = 0; j < ggml_parallel_devices.size(); j++) {
        ggml_tensor * wrapped = ggml_backend_tp_clone_tensor(tensor);
        extra->tensors[j] = wrapped;

        if (ctx->split) {
            extra->split_tensors = GGML_TP_SPLIT_ROWS;

            auto splits = get_row_splits(tensor);

            // these are weights which pretransposed and thus are split on rows
            wrapped->nb[2] = wrapped->nb[2] / wrapped->ne[1] * splits.split[j];
            wrapped->nb[3] = wrapped->nb[3] / wrapped->ne[1] * splits.split[j];

            // update row count
            wrapped->ne[1] = splits.split[j];
        }
    }

    ggml_backend_tp_finish_init_tensor(tensor);

    return GGML_STATUS_SUCCESS;
}


static void ggml_backend_tp_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_backend_set_tensor_async_common(buffer, tensor, data, offset, size);
}

static void ggml_backend_tp_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_tensor_parallel_extra * extra = (ggml_tensor_parallel_extra *)tensor->extra;

    for (size_t j = 0; j < ggml_parallel_devices.size(); j++) {
        auto be = ggml_parallel_backends[j];
        if (!be->iface.get_tensor_async) {
            ggml_backend_synchronize(be);
        }
    }

    ensure_rejoined(nullptr, tensor);
    rejoin_tensor(tensor, extra);

    if (extra->split_tensors) {
        // have never seen this called
        auto r = extra->converted_tensors[0];
        auto buft = r->buffer;
        buft->iface.get_tensor(buft, r, data, offset, size);
    }
    else {
        auto r = extra->tensors[0];
        auto buft = r->buffer;
        buft->iface.get_tensor(buft, r, data, offset, size);
    }

    GGML_UNUSED(buffer);
    GGML_UNUSED(tensor);
    GGML_UNUSED(offset);
    GGML_UNUSED(size);
}

static void ggml_backend_tp_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_tp_buffer_context * ctx = (ggml_backend_tp_buffer_context *)buffer->context;
    for (size_t j = 0; j < ggml_parallel_devices.size(); j++) {
        auto backend_buffer = ctx->backend_buffers[j];
        backend_buffer->iface.clear(backend_buffer, value);
    }
    GGML_UNUSED(value);
}

static void ggml_backend_tp_buffer_reset(ggml_backend_buffer_t buffer) {
    ggml_backend_tp_buffer_context * ctx = (ggml_backend_tp_buffer_context *)buffer->context;
    ctx->reset();
    GGML_UNUSED(buffer);
}

static ggml_backend_buffer_i ggml_backend_tp_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_tp_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_tp_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_tp_buffer_init_tensor,
    /* .memset_tensor   = */ NULL,
    /* .set_tensor      = */ ggml_backend_tp_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_tp_buffer_get_tensor,
    /* .cpy_tensor      = */ NULL,
    /* .clear           = */ ggml_backend_tp_buffer_clear,
    /* .reset           = */ ggml_backend_tp_buffer_reset,
};

static ggml_backend_buffer_t ggml_backend_tp_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_tp_buffer_context * ctx = new ggml_backend_tp_buffer_context();
    ctx->base_ptr = (void *) ggml_backend_tp_buffer_type_get_alignment(buft);
    ctx->split = ggml_backend_buft_is_tp_split(buft);

    if (ctx->split) {
        ggml_split splits = get_alloc_splits(size);

        for (size_t j = 0; j < ggml_parallel_devices.size(); j++) {
            auto buffer_type = ggml_parallel_devices[j]->iface.get_buffer_type(ggml_parallel_devices[j]);
            ctx->backend_buffers[j] = buffer_type->iface.alloc_buffer(buffer_type, splits.split[j]);
        }
    }
    else {
        for (size_t j = 0; j < ggml_parallel_devices.size(); j++) {
            auto buffer_type = ggml_parallel_devices[j]->iface.get_buffer_type(ggml_parallel_devices[j]);
            ctx->backend_buffers[j] = buffer_type->iface.alloc_buffer(buffer_type, size);
        }
    }
    return ggml_backend_buffer_init(buft, ggml_backend_tp_buffer_interface, ctx, size);
}


static size_t ggml_backend_tp_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    GGML_UNUSED(buft);
    GGML_UNUSED(tensor);

    size_t max_alloc_size = 0;
    for (size_t i = 0; i < ggml_parallel_devices.size(); i++) {
        auto dev = ggml_parallel_devices[i];
        auto buffer_type = dev->iface.get_buffer_type(dev);
        auto alloc_size = buffer_type->iface.get_alloc_size(buffer_type, tensor);
        max_alloc_size = std::max(max_alloc_size, alloc_size);
    }

    // to get cleanly diviible splits, make sure the allocation alignment is the multiple of the number of devices
    max_alloc_size = ggml_align_size(max_alloc_size, ggml_backend_tp_buffer_type_get_alignment(buft) * ggml_parallel_devices.size());
    return max_alloc_size;
}

static ggml_backend_buffer_type_i ggml_backend_tp_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_tp_buffer_type_name,
    /* .alloc_buffer     = */ ggml_backend_tp_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_tp_buffer_type_get_alignment,
    /* .get_max_size     = */ NULL,
    /* .get_alloc_size   = */ ggml_backend_tp_buffer_type_get_alloc_size,
    /* .is_host          = */ NULL,
};

ggml_backend_buffer_type_t ggml_backend_tp_buffer_type() {
    static ggml_backend_tp_buffer_type_context * buft_ctx = new ggml_backend_tp_buffer_type_context {
        /* .split = */ false,
    };

    static ggml_backend_buffer_type_t buft = new ggml_backend_buffer_type {
        /* .iface   = */ ggml_backend_tp_buffer_type_interface,
        /* .device  = */ ggml_backend_tp_reg_get_device(ggml_backend_tp_reg(), 0),
        /* .context = */ buft_ctx
    };
    return buft;
}

static ggml_backend_t ggml_backend_tp_device_init(ggml_backend_dev_t dev, const char * params) {
    ggml_backend_tp_device_context * ctx = (ggml_backend_tp_device_context *)dev->context;

    return ggml_backend_tp_init(ctx->endpoint.c_str());

    GGML_UNUSED(params);
}

static ggml_backend_buffer_type_t ggml_backend_tp_device_get_buffer_type(ggml_backend_dev_t dev) {
    return ggml_backend_tp_buffer_type();

    GGML_UNUSED(dev);
}

static bool ggml_backend_tp_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    GGML_UNUSED(dev);
    GGML_UNUSED(op);

    auto buft = op->buffer ? op->buffer->buft : nullptr;
    if (buft && (!ggml_backend_buft_is_tp_split(buft) && !ggml_backend_buft_is_tp(buft))) {
        return false;
    }

    // the tensor must also be compatible with all the parallel devices.
    for (size_t i = 0; i < ggml_parallel_devices.size(); i++) {
        auto dev = ggml_parallel_devices[i];
        if (!dev->iface.supports_op(dev, op)) {
            return false;
        }
    }

    if (op->op != GGML_OP_MUL_MAT && op->op != GGML_OP_MUL_MAT_ID) {
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            auto src = op->src[i];
            if (!src) {
                break;
            }
            if (src->buffer && ggml_backend_buft_is_tp_split(src->buffer->buft)) {
                return false;
            }
        }

        return true;
    }

    // only src0 is supported for split buffer. src1 is never called here with a weight buffer,
    // so this path is never actually used, only for sanity checking.
    auto src1 = op->src[1];
    if (src1->buffer && ggml_backend_buft_is_tp_split(src1->buffer->buft)) {
        return false;
    }

    auto src0 = op->src[0];
    if (!ggml_backend_buft_is_tp_split(src0->buffer->buft)) {
        return true;
    }

    // using something too small reduces performance due to additional rejoins.
    // return src0->ne[1] >= 2048;
    return src0->ne[1] >= 4096;
    return src0->ne[1] >= 8192;
}

static bool ggml_backend_tp_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    if (!buft) {
        return false;
    }

    if (buft->iface.get_name == ggml_backend_tp_buffer_type_name) {
        return true;
    }

    if (buft->iface.get_name == ggml_backend_tp_split_buffer_type_name) {
        return true;
    }

    return false;

    GGML_UNUSED(dev);
}

static const struct ggml_backend_device_i ggml_backend_tp_device_i = {
    /* .get_name             = */ ggml_backend_tp_device_get_name,
    /* .get_description      = */ ggml_backend_tp_device_get_description,
    /* .get_memory           = */ ggml_backend_tp_device_get_memory,
    /* .get_type             = */ ggml_backend_tp_device_get_type,
    /* .get_props            = */ ggml_backend_tp_device_get_props,
    /* .init_backend         = */ ggml_backend_tp_device_init,
    /* .get_buffer_type      = */ ggml_backend_tp_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ NULL,
    /* .supports_op          = */ ggml_backend_tp_device_supports_op,
    /* .supports_buft        = */ ggml_backend_tp_device_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

// backend reg interface

static const char * ggml_backend_tp_reg_get_name(ggml_backend_reg_t reg) {
    return "TensorParallel";

    GGML_UNUSED(reg);
}

static size_t ggml_backend_tp_reg_get_device_count(ggml_backend_reg_t reg) {
    return NUM_DEVICES;

    GGML_UNUSED(reg);
}

static ggml_backend_dev_t ggml_backend_tp_reg_get_device(ggml_backend_reg_t reg, size_t index) {

    static ggml_backend_tp_device_context * dev_ctx = new ggml_backend_tp_device_context;
    dev_ctx->name = "TensorParallel";

    static ggml_backend_dev_t dev = new ggml_backend_device {
        /* .iface   = */ ggml_backend_tp_device_i,
        /* .reg     = */ reg,
        /* .context = */ dev_ctx
    };

    return dev;
    
    GGML_ABORT("The tp backend does not have enumerated devices - use ggml_backend_add_device instead");

    GGML_UNUSED(reg);
    GGML_UNUSED(index);
}

static bool ggml_backend_tp_set_backends(std::vector<ggml_backend_reg_t> & backends) {
    if (!NUM_DEVICES) {
        return false;
    }

    for (auto backend : backends) {
        for (size_t i = 0; i < backend->iface.get_device_count(backend); i++) {
            auto device = backend->iface.get_device(backend, i);
            ggml_parallel_devices.push_back(device);
            printf("ggml_backend_tp_reg: registered device %s (%s)\n",
                ggml_backend_dev_name(device), ggml_backend_dev_description(device));

            auto be = device->iface.init_backend(device, nullptr);
            ggml_parallel_backends.push_back(be);
        }
    }

    ggml_backend_tp_threadpool_init(&ggml_device_threadpool, ggml_parallel_devices.size());
    
    return true;
}

static ggml_backend_buffer_type_i ggml_backend_tp_split_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_tp_split_buffer_type_name,
    /* .alloc_buffer     = */ ggml_backend_tp_buffer_type_interface.alloc_buffer,
    /* .get_alignment    = */ ggml_backend_tp_buffer_type_interface.get_alignment,
    /* .get_max_size     = */ NULL,
    /* .get_alloc_size   = */ ggml_backend_tp_buffer_type_interface.get_alloc_size,
    /* .is_host          = */ NULL,
};

static ggml_backend_buffer_type_t ggml_tp_split_buffer_type(int main_device, const float * tensor_split) {
    auto * ctx = new ggml_backend_tp_buffer_type_context {
        /* .split = */ true,
    };

    static struct ggml_backend_buffer_type buft {
        /* .iface   = */ ggml_backend_tp_split_buffer_type_interface,
        /* .device  = */ ggml_backend_tp_reg_get_device(ggml_backend_tp_reg(), 0),
        /* .context = */ ctx,
    };

    return &buft;

    GGML_UNUSED(main_device);
    GGML_UNUSED(tensor_split);
}

static void * ggml_backend_tp_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    if (std::strcmp(name, "ggml_backend_tp_set_backends") == 0) {
        return (void *)ggml_backend_tp_set_backends;
    }
    if (strcmp(name, "ggml_backend_split_buffer_type") == 0) {
        return (void *)ggml_tp_split_buffer_type;
    }

    return NULL;

    GGML_UNUSED(reg);
    GGML_UNUSED(name);
}

static const struct ggml_backend_reg_i ggml_backend_tp_reg_i = {
    /* .get_name         = */ ggml_backend_tp_reg_get_name,
    /* .get_device_count = */ ggml_backend_tp_reg_get_device_count,
    /* .get_device       = */ ggml_backend_tp_reg_get_device,
    /* .get_proc_address = */ ggml_backend_tp_get_proc_address,
};

ggml_backend_reg_t ggml_backend_tp_reg(void) {
    static struct ggml_backend_reg ggml_backend_tp_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_tp_reg_i,
        /* .context     = */ NULL,
    };

    return &ggml_backend_tp_reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_tp_reg)
