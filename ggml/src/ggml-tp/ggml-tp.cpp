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

// #define GGML_BACKEND_TP_VALIDATE 1
#define NUM_DEVICES 1
#define GGML_BACKEND_TP_TENSOR_CPY

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

#define TP_MAX_DEVICES 16

struct ggml_tensor_parallel_extra {
    // these are the tensors that are on the parallel GPU, they may or may not be split.
    ggml_tensor * tensors[TP_MAX_DEVICES];
    // flag for whether the tensors are split
    bool split_tensors;

#if GGML_BACKEND_TP_VALIDATE
    char *original_data;
    size_t original_size;
#endif

    bool computed;

    // persist the split tensor memory layout for reshaped tensors
    int64_t original_ne1;
    size_t original_nb1;

    // this tensor needs to be rejoined for another op.
    bool has_rejoin;
    // this tensor needs to be split for another op.
    bool has_split;

    // this tensor does not support split ops and has a src that needs to be rejoined.
    bool needs_src_rejoin;

    // this tensor has been rejoined already for this forward pass.
    // this property is reset every forward pass.
    bool rejoined;
    char * rejoined_buffer;
    size_t rejoined_size;

    size_t rejoined_buft_offsets[TP_MAX_DEVICES];

    // holds either split tensors or rejoined tensors depending on the owning tensor.
    ggml_tensor * converted_tensors[TP_MAX_DEVICES];

    // when a tensor is rejoined, it gathers the data from all the devices.
    // each device will provide a view of their full data for other devices as copy destinations.
    ggml_tensor * rejoined_tensor_views[TP_MAX_DEVICES][TP_MAX_DEVICES];

    ~ggml_tensor_parallel_extra() {
        if (rejoined_buffer) {
            free(rejoined_buffer);
            rejoined_buffer = nullptr;
        }

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

static void unwrap_tensor(ggml_tensor * tensor, std::set<ggml_tensor *> & tensors) {
    auto found = tensors.find(tensor);
    if (found != tensors.end()) {
        return;
    }

    ggml_tensor_parallel_extra * extra = (ggml_tensor_parallel_extra *)tensor->extra;
    tensors.insert(tensor);

    for (int i = 0; i < GGML_MAX_SRC; i++) {
        auto src = tensor->src[i];
        if (!src) {
            continue;
        }

        unwrap_tensor(src, tensors);
        auto src_extra = (ggml_tensor_parallel_extra *)src->extra;

        for (size_t j = 0; j < ggml_parallel_devices.size(); j++) {
            auto wrapped = extra->tensors[j];

            if (extra->split_tensors) {
                if (src_extra->split_tensors) {
                    wrapped->src[i] = src_extra->tensors[j];
                }
                else {
                    wrapped->src[i] = src_extra->converted_tensors[j];
                }
            }
            else {
                if (src_extra->split_tensors) {
                    wrapped->src[i] = src_extra->converted_tensors[j];
                }
                else {
                    wrapped->src[i] = src_extra->tensors[j];
                }
            }

            // special case for matmul
            if (i == 1 && tensor->op == GGML_OP_MUL_MAT) {
                auto src1 = tensor->src[1];
                auto src1_extra = (ggml_tensor_parallel_extra *)src1->extra;
                if (src1_extra->split_tensors) {
                    wrapped->src[1] = src1_extra->converted_tensors[j];
                }
                else {
                    wrapped->src[1] = src1_extra->tensors[j];
                }
            }

            if (!wrapped->src[i]) {
                GGML_LOG_ERROR("Tensor %s is not split, but its source %s is split\n", tensor->name, src->name);
            }
        }
    }
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
    return extra->split_tensors;
}

static bool is_split_compatible(ggml_tensor * tensor) {
    auto op = tensor->op;
    if (op == GGML_OP_MUL_MAT) {
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
        case GGML_OP_ADD:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
        // case GGML_OP_RESHAPE:
        // case GGML_OP_ROPE:
        // case GGML_OP_CPY:
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

static ggml_status ensure_split(const ggml_tensor *src) {
    auto src_extra = (ggml_tensor_parallel_extra *)src->extra;
    if (src_extra->split_tensors) {
        // this tensor is already split, so we don't need to do anything
        return GGML_STATUS_SUCCESS;
    }
    if (src_extra->has_split) {
        return GGML_STATUS_SUCCESS;
    }
    src_extra->has_split = true;

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
    void * base_ptr;
    ggml_backend_buffer_t backend_buffers[TP_MAX_DEVICES];
    bool split = false;
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
            if (backend_buffer) {
                if (backend_buffer && backend_buffer->iface.reset) {
                    backend_buffer->iface.reset(backend_buffer);
                }
            }

            rejoined_buft_sizes[i] = 0;
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

static ggml_status ensure_rejoined(const ggml_tensor *reason, const ggml_tensor * src) {
    auto src_extra = (ggml_tensor_parallel_extra *)src->extra;
    if (!src_extra->split_tensors) {
        // this tensor is not split, so we can't rejoin it
        return GGML_STATUS_SUCCESS;
    }

    auto reason_extra = (ggml_tensor_parallel_extra *)reason->extra;
    if (!reason_extra->needs_src_rejoin) {
        reason_extra->needs_src_rejoin = true;
    }

    if (src_extra->has_rejoin) {
        return GGML_STATUS_SUCCESS;
    }
    src_extra->has_rejoin = true;

    // printf("Rejoining tensor for %s %d %d\n", ggml_op_name(reason->op), src->ne[0], src->ne[1]);

    const auto alignment = ggml_backend_tp_buffer_type_get_alignment(src->buffer->buft);

    auto ctx = (ggml_backend_tp_buffer_context *)src->buffer->context;

    for (size_t j = 0; j < ggml_parallel_devices.size(); j++) {
        auto dev = ggml_parallel_devices[j];
        auto buffer_type = dev->iface.get_buffer_type(dev);

        auto tensor_size = buffer_type->iface.get_alloc_size(buffer_type, src);
        auto aligned_size = ggml_align_size(tensor_size, alignment);

        ggml_tensor * rejoined = ggml_backend_tp_clone_tensor(src);
        src_extra->converted_tensors[j] = rejoined;
        src_extra->rejoined_buft_offsets[j] = ctx->rejoined_buft_sizes[j];
        ctx->rejoined_buft_sizes[j] += aligned_size;

        // since this is an input, rewrite the op.
        rejoined->op = GGML_OP_NONE;
    }

    auto splits = get_col_splits(src);

    // a tensor that is split across 4 devices can not be concatenated memorywise (rowwise).
    // rather, the tensors must be copied back in columnwise.
    // a 4x4 tensor with 4 GPU holding 4x2 tensors each:
    // A A B B C C D D
    // A A B B C C D D
    // A A B B C C D D
    // A A B B C C D D
    for (size_t j = 0; j < ggml_parallel_devices.size(); j++) {
        auto rejoined = src_extra->converted_tensors[j];

        size_t col_offset = 0;
        for (size_t i = 0; i < ggml_parallel_devices.size(); i++) {
            auto view = ggml_backend_tp_clone_tensor(rejoined);
            src_extra->rejoined_tensor_views[j][i] = view;

            view->view_src = rejoined;
            view->ne[0] = splits.split[i];
            // adjust the offset to the start of the column in the destination tensor
            view->view_offs = rejoined->nb[1] / rejoined->ne[0] * col_offset;

            col_offset += splits.split[j];
        }
    }

    return GGML_STATUS_SUCCESS;

    GGML_UNUSED(reason);
}

static int memdiff_index(const char *a, const char *b, size_t length) {
    for (size_t i = 0; i < length; ++i) {
        if (a[i] != b[i]) {
            return (int)i;  // return index of first difference
        }
    }
    return -1;  // no differences found
}

static void read_tensor(ggml_tensor *tensor, std::unique_ptr<char, decltype(&std::free)> & memory) {
    auto tensor_size = ggml_nbytes(tensor);
    auto buffer = tensor->buffer;

    memory.reset((char *) malloc(tensor_size));

    buffer->iface.get_tensor(buffer, tensor, memory.get(), 0, tensor_size);
}

static void rejoin_tensor_data(const ggml_tensor * tensor, ggml_tensor_parallel_extra * extra, char *data) {
    auto recombined_size = ggml_nbytes(tensor);

    if (!extra->split_tensors) {
        return;
    }

    if (extra->rejoined) {
        return;
    }

    auto ne1 = extra->original_ne1 ? extra->original_ne1 : tensor->ne[1];
    auto nb1 = extra->original_nb1 ? extra->original_nb1 : tensor->nb[1];
    auto split_nb1 = get_dim_splits(nb1);

    if (data && extra->rejoined && extra->rejoined_buffer) {
        memcpy(data, extra->rejoined_buffer, recombined_size);
        return;
    }

    if (!data) {
        if (!extra->rejoined_buffer || extra->rejoined_size < recombined_size) {
            if (extra->rejoined_buffer) {
                free(extra->rejoined_buffer);
                extra->rejoined_buffer = nullptr;
            }
            extra->rejoined_buffer = (char *) malloc(recombined_size);
            extra->rejoined_size = recombined_size;
        }
    }
    auto rejoined_buffer = data ? data : extra->rejoined_buffer;

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

    extra->rejoined = true;

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
    for (int i = 0; i < thread->peers->size(); i++) {
        auto t = thread->peers->at(i);
        ggml_backend_tp_semaphore_release(&t->semaphore, 1);
    }
}

static void ggml_backend_tp_buffer_graph_compute_one(struct compute_thread * thread) {
    auto startTime = std::chrono::high_resolution_clock::now();

    auto cgraph = thread->cgraph;
    auto device_index = thread->device_index;
    for (int node_index = 0; node_index < cgraph->n_nodes; node_index++) {
        auto tensor = cgraph->nodes[node_index];
        auto extra = (ggml_tensor_parallel_extra *)tensor->extra;

        auto wrapped = extra->tensors[device_index];
        auto be = ggml_parallel_backends[device_index];
        ggml_status status = be->iface.node_compute(be, wrapped);
        if (status != GGML_STATUS_SUCCESS) {
            thread->end = status;
            release_peers(thread);
            thread->done.unlock();
            return;
        }

        extra->computed = true;

        // // this split op needs to be recombined for the another op
        if (extra->has_rejoin) {
            // copy this tensor to rejoined tensor views on all other devices.

#ifdef GGML_BACKEND_TP_TENSOR_CPY
            // async copies
            for (size_t other_device_index = 0; other_device_index < ggml_parallel_devices.size(); other_device_index++) {
                auto other_be = ggml_parallel_backends[other_device_index];
                auto rejoined_tensor_view = extra->rejoined_tensor_views[other_device_index][device_index];
                if (!rejoined_tensor_view) {
                    break;
                }

                if (!be->iface.cpy_tensor_async) {
                    continue;
                }

                if (!be->iface.cpy_tensor_async(be, other_be, wrapped, rejoined_tensor_view)) {
                    ggml_backend_tensor_copy(wrapped, rejoined_tensor_view);
                }
            }

            // sync copies
            for (size_t other_device_index = 0; other_device_index < ggml_parallel_devices.size(); other_device_index++) {
                auto other_be = ggml_parallel_backends[other_device_index];
                auto rejoined_tensor_view = extra->rejoined_tensor_views[other_device_index][device_index];
                if (!rejoined_tensor_view) {
                    break;
                }

                if (be->iface.cpy_tensor_async) {
                    continue;
                }

                ggml_backend_tensor_copy(wrapped, rejoined_tensor_view);
            }
#endif
            thread->end = node_index;
            release_peers(thread);

            // wait for everyone else
            for (size_t i = 0; i < thread->peers->size(); i++) {
                ggml_backend_tp_semaphore_acquire(&thread->semaphore);
            }

            #ifndef GGML_BACKEND_TP_TENSOR_CPY
            if (device_index == 0) {
                rejoin_tensor(tensor, extra);
                release_peers(thread);
            }

            ggml_backend_tp_semaphore_acquire(&thread->semaphore);
            #endif
        }
    }

    thread->end = cgraph->n_nodes;
    release_peers(thread);
    thread->done.unlock();

    // printf("TP graph %d compute time: %ld us\n", device_index, std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - startTime).count());
}

static enum ggml_status ggml_backend_tp_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    auto lastStartTime = std::chrono::high_resolution_clock::now();

    std::set<ggml_tensor*> tensors;

    for (int i = 0; i < cgraph->n_nodes; i++) {
        auto tensor = cgraph->nodes[i];
        ggml_tensor_parallel_extra * extra = (ggml_tensor_parallel_extra *)tensor->extra;
        // reset the rejoined state in case this tensor needs it.
        extra->rejoined = false;
        extra->computed = false;
        unwrap_tensor(tensor, tensors);
    }

    std::set<ggml_backend_tp_buffer_context *> contexts;
    for (auto tensor : tensors) {
        auto extra = (ggml_tensor_parallel_extra *)tensor->extra;
        if (tensor->view_src && extra->needs_src_rejoin) {
            auto view_src = tensor->view_src;
            auto view_src_extra = (ggml_tensor_parallel_extra *)view_src->extra;
            if (view_src_extra->has_rejoin) {
                for (size_t j = 0; j < ggml_parallel_devices.size(); j++) {
                    auto rejoined = view_src_extra->converted_tensors[j];
                    auto wrapped = extra->tensors[j];
                    wrapped->view_src = rejoined;
                    wrapped->view_offs = extra->rejoined_buft_offsets[j];
                    wrapped->data = (char *)rejoined->data + wrapped->view_offs;
                }
            }
        }
        if (!extra->has_rejoin) {
            continue;
        }

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
                rejoined_tensor_view->view_offs = extra->rejoined_buft_offsets[j];
            }
        }
    }

    std::map<std::string, int64_t> total_rejoin_times;

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

    printf("TP graph compute time: %ld us\n", std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - lastStartTime).count());
    lastStartTime = std::chrono::high_resolution_clock::now();


    return failed ? GGML_STATUS_FAILED : GGML_STATUS_SUCCESS;

    bool late_rejoin = false;

    for (int i = 0; i < cgraph->n_nodes; i++) {
        auto tensor = cgraph->nodes[i];
        auto extra = (ggml_tensor_parallel_extra *)tensor->extra;

        if (late_rejoin && extra->needs_src_rejoin) {
            for (size_t j = 0; j < ggml_parallel_devices.size(); j++) {
                auto be = ggml_parallel_backends[j];
                if (!be->iface.get_tensor_async) {
                    ggml_backend_synchronize(be);
                }
            }

            for (int j = 0; j < GGML_MAX_SRC; j++) {
                auto src = tensor->src[j];
                if (!src) {
                    continue;
                }
                auto src_extra = (ggml_tensor_parallel_extra *)src->extra;
                if (src_extra->has_rejoin) {
                    auto startTime = std::chrono::high_resolution_clock::now();
                    rejoin_tensor(src, src_extra);
                    auto time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - startTime).count();
                    total_rejoin_times[ggml_op_name(tensor->op)] += time;
                }
            }
        }

        for (size_t j = 0; j < ggml_parallel_devices.size(); j++) {
            auto wrapped = extra->tensors[j];
            auto be = ggml_parallel_backends[j];
            ggml_status status = be->iface.node_compute(be, wrapped);
            if (status != GGML_STATUS_SUCCESS) {
                return status;
            }
        }

        extra->computed = true;

        #ifdef GGML_BACKEND_TP_TENSOR_CPY
        if (extra->has_rejoin) {
            for (int device_index = 0; device_index < ggml_parallel_devices.size(); device_index++) {
                auto wrapped = extra->tensors[device_index];
                auto be = ggml_parallel_backends[device_index];

                // async copies
                for (size_t other_device_index = 0; other_device_index < ggml_parallel_devices.size(); other_device_index++) {
                    auto other_be = ggml_parallel_backends[other_device_index];
                    auto rejoined_tensor_view = extra->rejoined_tensor_views[other_device_index][device_index];
                    if (!rejoined_tensor_view) {
                        break;
                    }
    
                    if (!be->iface.cpy_tensor_async) {
                        continue;
                    }
    
                    if (!be->iface.cpy_tensor_async(be, other_be, wrapped, rejoined_tensor_view)) {
                        ggml_backend_tensor_copy(wrapped, rejoined_tensor_view);
                    }
                }
    
                // sync copies
                for (size_t other_device_index = 0; other_device_index < ggml_parallel_devices.size(); other_device_index++) {
                    auto other_be = ggml_parallel_backends[other_device_index];
                    auto rejoined_tensor_view = extra->rejoined_tensor_views[other_device_index][device_index];
                    if (!rejoined_tensor_view) {
                        break;
                    }
    
                    if (be->iface.cpy_tensor_async) {
                        continue;
                    }
    
                    ggml_backend_tensor_copy(wrapped, rejoined_tensor_view);
                }
            }
        }

        #else
        // // this split op needs to be recombined for the another op
        if (!late_rejoin && extra->has_rejoin) {
            for (size_t j = 0; j < ggml_parallel_devices.size(); j++) {
                auto be = ggml_parallel_backends[j];
                if (!be->iface.get_tensor_async) {
                    ggml_backend_synchronize(be);
                }
            }

            auto startTime = std::chrono::high_resolution_clock::now();
            rejoin_tensor(tensor, extra);
            auto time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - startTime).count();
            total_rejoin_times[ggml_op_name(tensor->op)] += time;
        }
        #endif

#if GGML_BACKEND_TP_VALIDATE
        auto src0 = tensor->src[0];
        char* original_data = NULL;
        size_t original_size = 0;
        ggml_tensor_parallel_extra * src0_extra = NULL;
        if (false && src0 && src0->extra) {
            src0_extra = (ggml_tensor_parallel_extra *)src0->extra;
            original_data = src0_extra->original_data;
            original_size = src0_extra->original_size;
        }
        if (original_data) {
            const int testindex = 1;

            auto buffer = extra->tensors[testindex]->buffer;
        
            auto buft = buffer->buft->iface.alloc_buffer(buffer->buft, original_size);
            ggml_tensor * test = ggml_backend_tp_clone_tensor(tensor->src[0]);
            test->buffer = buft;
            test->data = buft->iface.get_base(buft);
            auto status = ggml_backend_buffer_init_tensor(buft, test);
            if (status != GGML_STATUS_SUCCESS) {
                GGML_LOG_ERROR("ggml_backend_tp_buffer_set_tensor: init_tensor failed for tensor %s\n", tensor->name);
            }
            // set the tensor
            buft->iface.set_tensor(buft, test, original_data, 0, original_size);
            
            for (auto be : ggml_parallel_backends) {
                ggml_backend_synchronize(be);
            }
        
            auto op = ggml_backend_tp_clone_tensor(tensor);
            op->src[0] = test;
            op->src[1] = ((ggml_tensor_parallel_extra *)tensor->src[1]->extra)->tensors[testindex];
            auto buft2 = buffer->buft->iface.alloc_buffer(buffer->buft, ggml_nbytes(op));
            op->buffer = buft2;
            op->data = buft2->iface.get_base(buft2);
            status = buft2->iface.init_tensor(buft2, op);
            if (status != GGML_STATUS_SUCCESS) {
                GGML_LOG_ERROR("ggml_backend_tp_buffer_set_tensor: init_tensor failed for tensor %s\n", tensor->name);
            }
            
            for (auto be : ggml_parallel_backends) {
                ggml_backend_synchronize(be);
            }
        
            // compute
            auto be = ggml_parallel_backends[testindex];
            status = be->iface.node_compute(be, op);
            if (status != GGML_STATUS_SUCCESS) {
                GGML_LOG_ERROR("ggml_backend_tp_buffer_set_tensor: node_compute failed for tensor %s\n", tensor->name);
            }
            
            for (auto be : ggml_parallel_backends) {
                ggml_backend_synchronize(be);
            }
        
            ensure_rejoined(tensor, tensor);
            auto recombined_size = ggml_nbytes(tensor);
            std::unique_ptr<char, decltype(&std::free)> data(
                static_cast<char*>(std::malloc(recombined_size)), &std::free);
            rejoin_tensor(tensor, extra);
            read_tensor(extra->converted_tensors[0], data);

            std::unique_ptr<char, decltype(&std::free)> data2(nullptr, &std::free);
            read_tensor(op, data2);
            auto r = memdiff_index(data.get(), data2.get(), ggml_nbytes(op));

            buft->iface.free_buffer(buft);
            buft2->iface.free_buffer(buft2);

            delete test;
            delete op;

            if (r != -1) {
                GGML_LOG_ERROR("ggml_backend_tp_buffer_set_tensor: data mismatch for tensor %s\n", tensor->name);
        
                data2.reset((char *)malloc(ggml_nbytes(src0)));
                // read it back
                size_t offset = 0;
                for (size_t j = 0; j < ggml_parallel_devices.size(); j++) {
                    auto wrapped = src0_extra->tensors[j];
                    auto buft = wrapped->buffer;
                    buft->iface.get_tensor(buft, wrapped, data2.get() + offset, 0, ggml_nbytes(wrapped));
                    offset += ggml_nbytes(wrapped);
                }    
                for (auto be : ggml_parallel_backends) {
                    ggml_backend_synchronize(be);
                }
                r = memdiff_index(data2.get(), src0_extra->original_data, ggml_nbytes(src0));
                if (r != -1) {
                    GGML_LOG_ERROR("ggml_backend_tp_buffer_set_tensor: data mismatch for tensor %s\n", tensor->name);
                }
        
                auto e = (ggml_tensor_parallel_extra *)tensor->src[1]->extra;
                std::unique_ptr<char, decltype(&std::free)> t1(nullptr, &std::free);
                std::unique_ptr<char, decltype(&std::free)> t2(nullptr, &std::free);
        
                read_tensor(e->tensors[0], t1);
                read_tensor(e->tensors[1], t2);
        
                r = memdiff_index(t1.get(), t2.get(), ggml_nbytes(e->tensors[0]));
                if (r != -1) {
                    GGML_LOG_ERROR("ggml_backend_tp_buffer_set_tensor: data mismatch for tensor %s\n", tensor->name);
                }
            }
        }


        if (!extra->split_tensors || extra->has_rejoin) {
            std::unique_ptr<char, decltype(&std::free)> t1(nullptr, &std::free);
            std::unique_ptr<char, decltype(&std::free)> t2(nullptr, &std::free);
            
            read_tensor(extra->tensors[0], t1);
            read_tensor(extra->tensors[1], t2);

            auto fail = false;
            auto r = memdiff_index(t1.get(), t2.get(), ggml_nbytes(extra->tensors[0]));
            if (extra->split_tensors && r == -1) {
                // split tensors should be different.
                GGML_LOG_ERROR("ggml_backend_tp_buffer_set_tensor: data mismatch for tensor %s\n", tensor->name);
                // fail = true;
            }
            else if (!extra->split_tensors && r != -1) {
                // non split tensors should be the same
                GGML_LOG_ERROR("ggml_backend_tp_buffer_set_tensor: data mismatch for tensor %s\n", tensor->name);
                fail = true;
            }

            if (extra->has_rejoin) {
                read_tensor(extra->has_rejoin ? extra->converted_tensors[0] : extra->tensors[0], t1);
                read_tensor(extra->has_rejoin ? extra->converted_tensors[1] : extra->tensors[1], t2);
                r = memdiff_index(t1.get(), t2.get(), ggml_nbytes(extra->tensors[0]));
                if (r != -1) {
                    GGML_LOG_ERROR("ggml_backend_tp_buffer_set_tensor: data mismatch for tensor %s\n", tensor->name);
                    fail = true;
                }
            }
            if (fail) {
                GGML_LOG_ERROR("ggml_backend_tp_buffer_set_tensor: data mismatch for tensor %s\n", tensor->name);
    
                for (int i = 0; i < GGML_MAX_SRC; i++) {
                    auto src = tensor->src[i];
                    if (src) {
                        auto src_extra = (ggml_tensor_parallel_extra *)src->extra;
                        read_tensor(src_extra->has_rejoin ? src_extra->converted_tensors[0] : src_extra->tensors[0], t1);
                        read_tensor(src_extra->has_rejoin ? src_extra->converted_tensors[1] : src_extra->tensors[1], t2);
                        auto r = memdiff_index(t1.get(), t2.get(), ggml_nbytes(src_extra->tensors[0]));
                        if (r != -1) {
                            GGML_LOG_ERROR("ggml_backend_tp_buffer_set_tensor: data mismatch for tensor %s\n", tensor->name);
                        }
                    }
                }

                std::vector<ggml_tensor *> srcs;
                for (size_t j = 0; j < ggml_parallel_devices.size(); j++) {
                    auto wrapped = extra->tensors[j];
                    auto be = ggml_parallel_backends[j];
                    ggml_status status = be->iface.node_compute(be, wrapped);
                    srcs.push_back(wrapped->src[0]);
                    if (status != GGML_STATUS_SUCCESS) {
                        return status;
                    }
                }

                {
                    read_tensor(srcs[0], t1);
                    read_tensor(srcs[1], t2);
                    auto r = memdiff_index(t1.get(), t2.get(), ggml_nbytes(srcs[0]));
                    if (r != -1) {
                        GGML_LOG_ERROR("ggml_backend_tp_buffer_set_tensor: data mismatch for tensor %s\n", tensor->name);
                    }
                }

                // this split op needs to be recombined for the another op
                if (extra->has_rejoin) {
                    rejoin_tensor(tensor, extra);
                }
                
                read_tensor(extra->has_rejoin ? extra->converted_tensors[0] : extra->tensors[0], t1);
                read_tensor(extra->has_rejoin ? extra->converted_tensors[1] : extra->tensors[1], t2);
                r = memdiff_index(t1.get(), t2.get(), ggml_nbytes(extra->tensors[0]));
                if (r != -1) {
                    GGML_LOG_ERROR("ggml_backend_tp_buffer_set_tensor: data mismatch for tensor %s\n", tensor->name);
                }
            }
        }
        else {
            printf("Tensor %s is not split\n", ggml_op_name(tensor->op));
        }
#endif
    }

    for (auto & rejoin_time : total_rejoin_times) {
        printf("Rejoin time for %s: %ld us\n", rejoin_time.first.c_str(), rejoin_time.second);
    }

    printf("TP graph compute time: %ld us\n", std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - lastStartTime).count());
    lastStartTime = std::chrono::high_resolution_clock::now();

    return GGML_STATUS_SUCCESS;

    GGML_UNUSED(backend);
}

static void ggml_backend_set_tensor_async_common(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_backend_tp_buffer_context * ctx = (ggml_backend_tp_buffer_context *)buffer->context;
    ggml_tensor_parallel_extra * extra = (ggml_tensor_parallel_extra *)tensor->extra;

    if (ctx->split) {
        if (tensor->ne[1] <= 1) {
            GGML_LOG_ERROR("ggml_backend_tp_buffer_set_tensor: tensor %s is not a split compatible tensor\n", tensor->name);
            return;
        }
        if (tensor->ne[2] > 1) {
            GGML_LOG_ERROR("ggml_backend_tp_buffer_set_tensor: tensor %s is not a split compatible tensor\n", tensor->name);
            return;
        }
        if (tensor->ne[3] > 1) {
            GGML_LOG_ERROR("ggml_backend_tp_buffer_set_tensor: tensor %s is not a split compatible tensor\n", tensor->name);
            return;
        }

        if (size % tensor->ne[1] != 0) {
            GGML_LOG_ERROR("ggml_backend_tp_buffer_set_tensor: size %zu is not a multiple of tensor->ne[1] %zu\n", size, tensor->ne[1]);
            return;
        }
        if (offset) {
            GGML_LOG_ERROR("ggml_backend_tp_buffer_set_tensor: offset %zu is not zero\n", offset);
            return;
        }
        if (tensor->view_src) {
            GGML_LOG_ERROR("ggml_backend_tp_buffer_set_tensor: split tensor %s has unsupported view_src %s\n", tensor->name, tensor->view_src->name);
            return;
        }

        // this should just be the value of tensor-nb[1] right?
        auto bytes_per_row = size / tensor->ne[1];
        if (bytes_per_row != tensor->nb[1]) {
            GGML_LOG_ERROR("ggml_backend_tp_buffer_set_tensor: bytes_per_row %zu != tensor->nb[1] %zu\n", bytes_per_row, tensor->nb[1]);
            return;
        }

        // weight matrices are transposed, so split on row
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

            auto split_offset = cur_row * bytes_per_row;
            auto split_size = (size_t) splits.split[j] * bytes_per_row;
            
            if (be->iface.set_tensor_async) {
                be->iface.set_tensor_async(be, wrapped, (const char *) data + split_offset, 0, split_size);
            }
            else {
                auto backend_buffer = ctx->backend_buffers[j];
                backend_buffer->iface.set_tensor(backend_buffer, wrapped, (const char *) data + split_offset, 0, split_size);
            }

            cur_row += splits.split[j];
        }

#if GGML_BACKEND_TP_VALIDATE
        extra->original_data = (char *)malloc(size);
        memcpy(extra->original_data, data, size);
        extra->original_size = size;
#endif
    }
    else {
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

    ensure_rejoined(tensor, tensor);
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
    if (dst_extra->split_tensors) {
        GGML_LOG_WARN("Tensor %s is split, but set_tensor_async is called\n", dst->name);
        return false;
    }

    if (ggml_backend_buffer_is_host(src->buffer)) {
        ggml_backend_tp_set_tensor_async(backend_dst, dst, src->data, 0, ggml_nbytes(src));
        return true;
    }

    if (src->buffer->buft != ggml_backend_tp_buffer_type()) {
        return false;
    }

    auto src_extra = (ggml_tensor_parallel_extra *)src->extra;

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
    /* .node_compute            = */ NULL,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
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

static enum ggml_status ggml_backend_tp_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    const auto alignment = ggml_backend_tp_buffer_type_get_alignment(buffer->buft);
    ggml_backend_tp_buffer_context * ctx = (ggml_backend_tp_buffer_context *)buffer->context;

    ggml_tensor_parallel_extra * extra = new ggml_tensor_parallel_extra();
    ctx->extras.push_back(extra);

    tensor->extra = extra;

    
    // determine whether this tensor op results in a split output.
    // this may be due to the weights themselves being split, or the tensor being a result of
    // a split compatible operation on a split src tensor.
    auto split = ctx->split;
    auto split_from_src = false;

    // sanity check assertion. expecting weights to come in as GGML_OP_NONE
    // if (split && !is_split_compatible(tensor)) {
    //     GGML_LOG_ERROR("ggml_backend_tp_buffer_init_tensor: tensor op %s is not split compatible but is marked as such\n", ggml_op_name(tensor->op));
    //     return GGML_STATUS_FAILED;
    // }

    // check all src tensors to see if this tensor is split or the src tensors need a rejoin
    auto tensor_is_split_compatible = is_split_compatible(tensor);
    if (!tensor_is_split_compatible) {
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            auto src = tensor->src[i];
            if (!src) {
                continue;
            }
            if (ggml_backend_buft_is_tp_split(src->buffer->buft)) {
                GGML_LOG_ERROR("ggml_backend_tp_buffer_init_tensor: tensor %s is split but src %s is not split compatible\n", tensor->name, src->name);
                return GGML_STATUS_FAILED;
            }
            if (ensure_rejoined(tensor, src) != GGML_STATUS_SUCCESS) {
                return GGML_STATUS_FAILED;
            }
        }
    }
    else if (!split) {
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            auto src = tensor->src[i];
            if (!src) {
                continue;
            }
            if (ggml_backend_tp_is_split(src)) {
                split = true;
                split_from_src = true;
                break;
            }
        }

        // everything but mul mat needs to be split as well.
        // mulmat can handle the broadcast "A" tensor.
        if (split_from_src) {
            if (tensor->op == GGML_OP_MUL_MAT) {
                ensure_rejoined(tensor, tensor->src[1]);
            }
            else if (tensor->op == GGML_OP_RESHAPE) {
                auto src0 = tensor->src[0];
                auto src0_extra = (ggml_tensor_parallel_extra *)src0->extra;
                extra->original_ne1 = src0_extra->original_ne1 ? src0_extra->original_ne1 : src0->ne[1];
                extra->original_nb1 = src0_extra->original_nb1 ? src0_extra->original_nb1 : src0->nb[1];
            }
            else if (tensor->op != GGML_OP_UNARY) {
                for (int i = 0; i < GGML_MAX_SRC; i++) {
                    auto src = tensor->src[i];
                    if (!src) {
                        continue;
                    }
                    ensure_split(src);
                }
            }
        }
    }

    extra->split_tensors = split;

    for (size_t j = 0; j < ggml_parallel_devices.size(); j++) {
        ggml_tensor * wrapped = ggml_backend_tp_clone_tensor(tensor);
        auto backend_buffer = ctx->backend_buffers[j];
        wrapped->buffer = backend_buffer;

        auto device_alignment = ggml_backend_tp_buffer_type_get_alignment(backend_buffer->buft);

        if (split) {
            if (tensor->op == GGML_OP_CPY) {
                ggml_split splits = get_col_splits(tensor);
                // adjust the stride for the new row count
                wrapped->nb[1] = wrapped->nb[1] / wrapped->ne[0] * splits.split[j];
                wrapped->nb[2] = wrapped->nb[2] / wrapped->ne[0] * splits.split[j];
                wrapped->nb[3] = wrapped->nb[3] / wrapped->ne[0] * splits.split[j];

                // update col count
                wrapped->ne[0] = splits.split[j];
            }
            else {
                // when calculating splits on view tensors, need to get the original column partitioning
                auto vs = tensor;
                while (vs->view_src) {
                    vs = vs->view_src;
                }

                // according to ggml-cuda.h and from what i've seen, the weight matrices are transposed
                // for better memory layout, but the dst (this tensor) is not transposed.
                ggml_split splits = vs->op != GGML_OP_NONE ? get_col_splits(vs) : get_row_splits(vs);
                
                if (splits.split[j] == 0) {
                    GGML_LOG_ERROR("ggml_backend_tp_buffer_init_tensor: split tensor %s has zero size\n", tensor->name);
                    return GGML_STATUS_FAILED;
                }

                if (tensor->op == GGML_OP_NONE) {
                    // these are weights which pretransposed and thus are split on rows
                    wrapped->nb[2] = wrapped->nb[2] / wrapped->ne[1] * splits.split[j];
                    wrapped->nb[3] = wrapped->nb[3] / wrapped->ne[1] * splits.split[j];

                    // update row count
                    wrapped->ne[1] = splits.split[j];
                }
                else {
                    auto original_ne0 = wrapped->ne[0];
                    if (tensor->op == GGML_OP_RESHAPE) {
                        // ehhhhh i dunno man.
                        // 8192x2x1x1 -> 128x64x2x1
                        // 8192 is column partitioned into two 4096x2x1x1
                        // 4096 / 128 = 32 so actual reshape result is 128x32x2x1
                        // auto splits = get_col_splits(tensor);

                        auto src = tensor->src[0];
                        auto src_extra = (ggml_tensor_parallel_extra *)src->extra;
                        auto src_cols = src_extra->tensors[j]->ne[0];

                        if (src_cols > wrapped->ne[0]) {
                            auto original_ne1 = wrapped->ne[1];
                            wrapped->ne[1] = src_cols / wrapped->ne[0];
                            wrapped->ne[2] = wrapped->ne[2] * original_ne1 / wrapped->ne[1];
                        }
                        else {
                            auto original_ne0 = wrapped->ne[0];
                            wrapped->ne[0] = src_cols;
                            wrapped->ne[2] = wrapped->ne[2] * original_ne0 / wrapped->ne[0];
                        }

                        wrapped->nb[1] = wrapped->nb[1] / original_ne0 * splits.split[j];
                        wrapped->nb[2] = wrapped->nb[0] * wrapped->ne[1];
                        wrapped->nb[3] = wrapped->nb[2] * wrapped->ne[2];
                    }
                    else {
                        if (tensor->op == GGML_OP_ROPE && split_from_src) {
                            // should probably do this for all reshape + ops...
                            auto src = tensor->src[0];
                            auto src_extra = (ggml_tensor_parallel_extra *)src->extra;
                            wrapped->ne[0] = src_extra->tensors[j]->ne[0];
                            wrapped->ne[1] = src_extra->tensors[j]->ne[1];
                            wrapped->ne[2] = src_extra->tensors[j]->ne[2];
                            wrapped->ne[3] = src_extra->tensors[j]->ne[3];
                            wrapped->nb[0] = src_extra->tensors[j]->nb[0];
                            wrapped->nb[1] = src_extra->tensors[j]->nb[1];
                            wrapped->nb[2] = src_extra->tensors[j]->nb[2];
                            wrapped->nb[3] = src_extra->tensors[j]->nb[3];
                        }
                        else {
                            // update col count
                            wrapped->ne[0] = splits.split[j];
                            // adjust the stride for the new row count
                            wrapped->nb[1] = wrapped->nb[1] / original_ne0 * splits.split[j];
                            wrapped->nb[2] = wrapped->nb[2] / original_ne0 * splits.split[j];
                            wrapped->nb[3] = wrapped->nb[3] / original_ne0 * splits.split[j];
                        }
                    }
                }
            }

        }

        if (!tensor->view_src) {
            // the virtual address of the buffer starts at the alignment value, so subtract that.
            auto tensor_base = (uint64_t) tensor->data - alignment;
            // tensor data is expected to be aligned.
            if (tensor_base % alignment) {
                GGML_LOG_ERROR("ggml_backend_tp_buffer_init_tensor: tensor %s is not aligned to %zu\n", tensor->name, alignment);
                return GGML_STATUS_FAILED;
            }
            auto tensor_blocks = tensor_base / alignment;

            if (tensor_blocks % ggml_parallel_devices.size()) {
                GGML_LOG_WARN("ggml_backend_tp_buffer_init_tensor: tensor %s is not aligned to device count %zu\n", tensor->name, alignment);
                return GGML_STATUS_FAILED;
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
                GGML_LOG_ERROR("ggml_backend_tp_buffer_init_tensor: split buffer type %s does not support views\n", buffer->buft->iface.get_name(buffer->buft));
                return GGML_STATUS_FAILED;
            }
            auto view_src_extra = (ggml_tensor_parallel_extra *)tensor->view_src->extra;
            auto view_src = view_src_extra->tensors[j];
            if (!tensor_is_split_compatible && view_src_extra->split_tensors) {
                ensure_rejoined(tensor, tensor->view_src);
                // the real view src is going to be late init for this.
            }
            auto rem = tensor->view_offs % alignment;
            auto view_offs = tensor->view_offs / alignment * device_alignment + rem;
            wrapped->data = (char *) view_src->data + view_offs;
            wrapped->view_src = view_src;
            wrapped->view_offs = view_offs;
            if (wrapped->view_src == NULL) {
                GGML_LOG_ERROR("ggml_backend_tp_buffer_init_tensor: view_src is NULL for tensor %s\n", tensor->name);
                return GGML_STATUS_FAILED;
            }
        }

        extra->tensors[j] = wrapped;
        auto result = backend_buffer->iface.init_tensor(backend_buffer, wrapped);
        if (result != GGML_STATUS_SUCCESS) {
            GGML_LOG_ERROR("ggml_backend_tp_buffer_init_tensor: init_tensor failed for tensor %s\n", tensor->name);
            return result;
        }
    }

    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_tp_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_backend_set_tensor_async_common(buffer, tensor, data, offset, size);

    // for (size_t j = 0; j < ggml_parallel_devices.size(); j++) {
    //     auto be = ggml_parallel_backends[j];
    //     if (be->iface.set_tensor_async) {
    //         be->iface.synchronize(be);
    //     }
    // }
}

static void ggml_backend_tp_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_tensor_parallel_extra * extra = (ggml_tensor_parallel_extra *)tensor->extra;

    for (size_t j = 0; j < ggml_parallel_devices.size(); j++) {
        auto be = ggml_parallel_backends[j];
        if (!be->iface.get_tensor_async) {
            ggml_backend_synchronize(be);
        }
    }

    ensure_rejoined(tensor, tensor);
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
    // return ggml_nbytes(tensor);
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

    if (op->op != GGML_OP_MUL_MAT) {
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            auto src = op->src[i];
            if (!src) {
                continue;
            }
            if (src->buffer && ggml_backend_buft_is_tp_split(src->buffer->buft)) {
                return false;
            }
        }
    }
    else {
        // only src0 is supported for split buffer.
        auto src1 = op->src[1];
        if (src1->buffer && ggml_backend_buft_is_tp_split(src1->buffer->buft)) {
            return false;
        }

        // the tensor must also be compatible with all the parallel devices.
        for (size_t i = 0; i < ggml_parallel_devices.size(); i++) {
            auto dev = ggml_parallel_devices[i];
            if (!dev->iface.supports_op(dev, op)) {
                return false;
            }
        }

        return true;
    }


    // the tensor must also be compatible with all the parallel devices.
    for (size_t i = 0; i < ggml_parallel_devices.size(); i++) {
        auto dev = ggml_parallel_devices[i];
        if (!dev->iface.supports_op(dev, op)) {
            return false;
        }
    }

    return true;
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
