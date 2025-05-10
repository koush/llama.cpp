#include "ggml-tp.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"
#include "ggml-cpp.h"

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


static std::vector<ggml_backend_dev_t> ggml_parallel_devices;
static std::vector<ggml_backend_t> ggml_parallel_backends;

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
    char *original_data;
    size_t original_size;

    bool computed;

    // in case the tensors are split and need to be rejoined, they are stored here.
    bool needs_rejoin;
    bool rejoined;
    ggml_backend_buffer_t rejoined_bufts[TP_MAX_DEVICES];
    ggml_tensor * rejoined_tensors[TP_MAX_DEVICES];

    ~ggml_tensor_parallel_extra() {
        for (size_t i = 0; i < TP_MAX_DEVICES; i++) {
            auto tensor = tensors[i];
            if (tensor) {
                delete tensor;
                tensors[i] = nullptr;
            }

            auto rejoined_tensor = rejoined_tensors[i];
            if (rejoined_tensor) {
                delete rejoined_tensor;
                rejoined_tensors[i] = nullptr;
            }

            auto rejoined_buft = rejoined_bufts[i];
            if (rejoined_buft) {
                rejoined_buft->iface.free_buffer(rejoined_buft);
                rejoined_bufts[i] = nullptr;
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
    GGML_UNUSED(backend);
    // this is no-op because we don't have any async operations
}

static void unwrap_tensor(ggml_tensor * tensor, std::map<ggml_tensor *, ggml_tensor_parallel_extra*> & tensor_map) {
    auto found = tensor_map.find(tensor);
    if (found != tensor_map.end()) {
        return;
    }

    ggml_tensor_parallel_extra * extra = (ggml_tensor_parallel_extra *)tensor->extra;
    tensor_map[tensor] = extra;

    for (int i = 0; i < GGML_MAX_SRC; i++) {
        auto src = tensor->src[i];
        if (!src) {
            continue;
        }

        unwrap_tensor(src, tensor_map);
        auto src_extra = (ggml_tensor_parallel_extra *)src->extra;

        for (size_t j = 0; j < ggml_parallel_devices.size(); j++) {
            auto wrapped = extra->tensors[j];

            if (!src_extra->split_tensors || extra->split_tensors) {
                wrapped->src[i] = src_extra->tensors[j];
            }
            else {
                if (!src_extra->needs_rejoin) {
                    GGML_LOG_ERROR("Tensor %s is not split, but its source %s is split\n", tensor->name, src->name);
                }
                wrapped->src[i] = src_extra->rejoined_tensors[j];
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

    return false;
    switch (op) {
        case GGML_OP_UNARY:
        case GGML_OP_MUL_MAT:
        case GGML_OP_ADD:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
        case GGML_OP_NONE:
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

static ggml_split get_col_splits(ggml_tensor * tensor) {
    return get_dim_splits(tensor->ne[0]);
}

static ggml_split get_row_splits(ggml_tensor * tensor) {
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

static ggml_status ensure_rejoined(const ggml_tensor * src) {
    auto src_extra = (ggml_tensor_parallel_extra *)src->extra;
    if (!src_extra->split_tensors) {
        // this tensor is not split, so we don't need to rejoin it
        return GGML_STATUS_SUCCESS;
    }

    if (src_extra->needs_rejoin) {
        return GGML_STATUS_SUCCESS;
    }
    src_extra->needs_rejoin = true;

    const auto alignment = ggml_backend_tp_buffer_type_get_alignment(src->buffer->buft);

    for (size_t j = 0; j < ggml_parallel_devices.size(); j++) {
        auto dev = ggml_parallel_devices[j];
        auto buffer_type = dev->iface.get_buffer_type(dev);

        auto tensor_size = buffer_type->iface.get_alloc_size(buffer_type, src);
        auto aligned_size = ggml_align_size(tensor_size, alignment);

        auto buft = buffer_type->iface.alloc_buffer(buffer_type, aligned_size);
        auto base = (char *) buft->iface.get_base(buft);

        src_extra->rejoined_bufts[j] = buft;

        ggml_tensor * rejoined = ggml_backend_tp_clone_tensor(src);
        // since this is an input, rewrite the op.
        rejoined->op = GGML_OP_NONE;
        src_extra->rejoined_tensors[j] = rejoined;

        rejoined->buffer = buft;
        rejoined->data = base;

        auto result = buft->iface.init_tensor(buft, rejoined);
        if (result != GGML_STATUS_SUCCESS) {
            return result;
        }
    }

    return GGML_STATUS_SUCCESS;
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

static void rejoin_tensor(const ggml_tensor * tensor, ggml_tensor_parallel_extra * extra, char * data) {
    auto recombined_size = ggml_nbytes(tensor);

    if (!extra->split_tensors) {
        return;
    }

    if (extra->rejoined) {
        return;
    }
    extra->rejoined = true;

    for (auto be : ggml_parallel_backends) {
        ggml_backend_synchronize(be);
    }

    // a tensor that is split across 4 devices can not be concatenated memorywise (rowwise).
    // rather, the tensors must be copied back in columnwise.
    // a 4x4 tensor with 4 GPU holding 1x4 tensors each:
    // A B C D
    // A B C D
    // A B C D
    // A B C D
    for (int64_t row = 0; row < tensor->ne[1]; row++) {
        size_t offset = 0;
        for (size_t j = 0; j < ggml_parallel_devices.size(); j++) {
            auto wrapped = extra->tensors[j];
            auto buft = wrapped->buffer;
            auto wrapped_row_offset = wrapped->nb[1] * row;
            // todo use async version
            buft->iface.get_tensor(buft, wrapped, data + row * tensor->nb[1] + offset, wrapped_row_offset, wrapped->nb[1]);
            offset += wrapped->nb[1];
        }
    }

    for (auto be : ggml_parallel_backends) {
        ggml_backend_synchronize(be);
    }

    for (size_t j = 0; j < ggml_parallel_devices.size(); j++) {
        auto r = extra->rejoined_tensors[j];
        auto buft = r->buffer;
        // todo use async version
        buft->iface.set_tensor(buft, r, data, 0, recombined_size);
    }
    
    for (auto be : ggml_parallel_backends) {
        ggml_backend_synchronize(be);
    }
}

static enum ggml_status ggml_backend_tp_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    std::map<ggml_tensor*, ggml_tensor_parallel_extra *> tensor_map;

    for (int i = 0; i < cgraph->n_nodes; i++) {
        auto tensor = cgraph->nodes[i];
        ggml_tensor_parallel_extra * extra = (ggml_tensor_parallel_extra *)tensor->extra;
        // reset the rejoined state in case this tensor needs it.
        extra->rejoined = false;
        extra->computed = false;

        unwrap_tensor(tensor, tensor_map);
    }

    // auto recombines = 0;

    auto startTime = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < cgraph->n_nodes; i++) {
        auto tensor = cgraph->nodes[i];
        auto extra = (ggml_tensor_parallel_extra *)tensor->extra;

        for (size_t j = 0; j < ggml_parallel_devices.size(); j++) {
            auto wrapped = extra->tensors[j];
            auto be = ggml_parallel_backends[j];
            ggml_status status = be->iface.node_compute(be, wrapped);
            if (status != GGML_STATUS_SUCCESS) {
                return status;
            }
        }

        extra->computed = true;

        // this split op needs to be recombined for the another op
        if (extra->needs_rejoin) {
            auto recombined_size = ggml_nbytes(tensor);
            std::unique_ptr<char, decltype(&std::free)> recombined(
                static_cast<char*>(std::malloc(recombined_size)), &std::free);
            rejoin_tensor(tensor, extra, recombined.get());
        }

            
        for (auto be : ggml_parallel_backends) {
            ggml_backend_synchronize(be);
        }

        if (false) {
        //if (!extra->split_tensors || extra->needs_rejoin) {
            std::unique_ptr<char, decltype(&std::free)> t1(nullptr, &std::free);
            std::unique_ptr<char, decltype(&std::free)> t2(nullptr, &std::free);

            read_tensor(extra->needs_rejoin ? extra->rejoined_tensors[0] : extra->tensors[0], t1);
            read_tensor(extra->needs_rejoin ? extra->rejoined_tensors[1] : extra->tensors[1], t2);
            auto r = memdiff_index(t1.get(), t2.get(), ggml_nbytes(extra->tensors[0]));
            if (r != -1) {
                GGML_LOG_ERROR("ggml_backend_tp_buffer_set_tensor: data mismatch for tensor %s\n", tensor->name);
    
                for (int i = 0; i < GGML_MAX_SRC; i++) {
                    auto src = tensor->src[i];
                    if (src) {
                        auto src_extra = (ggml_tensor_parallel_extra *)src->extra;
                        read_tensor(src_extra->needs_rejoin ? src_extra->rejoined_tensors[0] : src_extra->tensors[0], t1);
                        read_tensor(src_extra->needs_rejoin ? src_extra->rejoined_tensors[1] : src_extra->tensors[1], t2);
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
                if (extra->needs_rejoin) {
                    auto recombined_size = ggml_nbytes(tensor);
                    std::unique_ptr<char, decltype(&std::free)> recombined(
                        static_cast<char*>(std::malloc(recombined_size)), &std::free);
                    rejoin_tensor(tensor, extra, recombined.get());
                }
                
                read_tensor(extra->needs_rejoin ? extra->rejoined_tensors[0] : extra->tensors[0], t1);
                read_tensor(extra->needs_rejoin ? extra->rejoined_tensors[1] : extra->tensors[1], t2);
                r = memdiff_index(t1.get(), t2.get(), ggml_nbytes(extra->tensors[0]));
                if (r != -1) {
                    GGML_LOG_ERROR("ggml_backend_tp_buffer_set_tensor: data mismatch for tensor %s\n", tensor->name);
                }
            }
        }
    }
    
    for (auto be : ggml_parallel_backends) {
        ggml_backend_synchronize(be);
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);

    // printf("TP graph compute unwrap time: %lld us\n", duration.count());

    return GGML_STATUS_SUCCESS;

    GGML_UNUSED(backend);
}

static ggml_backend_i ggml_backend_tp_interface = {
    /* .get_name                = */ ggml_backend_tp_name,
    /* .free                    = */ ggml_backend_tp_free,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
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

    for (size_t i = 0; i < ggml_parallel_devices.size(); i++) {
        auto dev = ggml_parallel_devices[i];
        auto be = dev->iface.init_backend(dev, nullptr);
        ggml_parallel_backends.push_back(be);
        // if (dev->reg == backend->reg) {
        //     backend->device = dev;
        //     break;
        // }
    }

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


struct ggml_backend_tp_buffer_context {
    void * base_ptr;
    ggml_backend_buffer_t backend_buffers[TP_MAX_DEVICES];
    bool split = false;
    std::vector<ggml_tensor_parallel_extra *> extras;

    ~ggml_backend_tp_buffer_context() {
        reset();

        for (size_t i = 0; i < TP_MAX_DEVICES; i++) {
            auto backend_buffer = backend_buffers[i];
            if (backend_buffer) {
                backend_buffer->iface.free_buffer(backend_buffer);
                backend_buffers[i] = nullptr;
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
        }

        for (auto & extra : extras) {
            delete extra;
        }
        extras.clear();
    }
};


static void ggml_backend_tp_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_tp_buffer_context * ctx = (ggml_backend_tp_buffer_context *)buffer->context;
    delete ctx;
}

static void * ggml_backend_tp_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_tp_buffer_context * ctx = (ggml_backend_tp_buffer_context *)buffer->context;
    return ctx->base_ptr;
}

static const char * ggml_backend_tp_buffer_type_name(ggml_backend_buffer_type_t buft) {
    return "TP";
    GGML_UNUSED(buft);
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
    // the virtual address of the buffer starts at the alignment value, so subtract that.
    auto tensor_base = (uint64_t) tensor->data - alignment;
    // tensor data is expected to be aligned.
    if (tensor_base % alignment) {
        GGML_LOG_ERROR("ggml_backend_tp_buffer_init_tensor: tensor %s is not aligned to %zu\n", tensor->name, alignment);
        return GGML_STATUS_FAILED;
    }
    auto tensor_blocks = tensor_base / alignment;
    if (tensor_blocks % ggml_parallel_devices.size()) {
        GGML_LOG_ERROR("ggml_backend_tp_buffer_init_tensor: tensor %s is not aligned to device count %zu\n", tensor->name, alignment);
        return GGML_STATUS_FAILED;
    }

    ggml_backend_tp_buffer_context * ctx = (ggml_backend_tp_buffer_context *)buffer->context;

    auto tensor_name = std::string(tensor->name);
    ggml_tensor_parallel_extra * extra = new ggml_tensor_parallel_extra();
    ctx->extras.push_back(extra);

    tensor->extra = extra;

    // according to ggml-cuda.h and from what i've seen, the weight matrices are transposed
    // for better memory layout, but the dst (this tensor) is not transposed.
    ggml_split splits = tensor->op != GGML_OP_NONE ? get_col_splits(tensor) : get_row_splits(tensor);
    
    // determine whether this tensor op results in a split output.
    // this may be due to the weights themselves being split, or the tensor being a result of
    // a split compatible operation on a split src tensor.
    auto split = ctx->split;

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
            auto src_extra = (ggml_tensor_parallel_extra *)src->extra;
            if (ggml_backend_buft_is_tp_split(src->buffer->buft) || src_extra->split_tensors) {
                if (ensure_rejoined(src) != GGML_STATUS_SUCCESS) {
                    return GGML_STATUS_FAILED;
                }
            }
        }
    }
    else if (!split) {
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            auto src = tensor->src[i];
            if (!src) {
                continue;
            }
            auto src_extra = (ggml_tensor_parallel_extra *)src->extra;
            if (ggml_backend_buft_is_tp_split(src->buffer->buft) || src_extra->split_tensors) {
                split = true;
            }
        }
    }

    extra->split_tensors = split;

    for (size_t j = 0; j < ggml_parallel_devices.size(); j++) {
        ggml_tensor * wrapped = ggml_backend_tp_clone_tensor(tensor);
        auto backend_buffer = ctx->backend_buffers[j];
        wrapped->buffer = backend_buffer;

        auto device_alignment = ggml_backend_tp_buffer_type_get_alignment(backend_buffer->buft);

        if (!tensor->view_src) {
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

            if (split) {
                if (splits.split[j] == 0) {
                    GGML_LOG_ERROR("ggml_backend_tp_buffer_init_tensor: split tensor %s has zero size\n", tensor->name);
                    return GGML_STATUS_FAILED;
                }

                if (tensor->op != GGML_OP_NONE) {
                    // adjust the stride for the new row count
                    wrapped->nb[1] = wrapped->nb[1] / wrapped->ne[0] * splits.split[j];
                    wrapped->nb[2] = wrapped->nb[2] / wrapped->ne[0] * splits.split[j];
                    wrapped->nb[3] = wrapped->nb[3] / wrapped->ne[0] * splits.split[j];

                    // update col count
                    wrapped->ne[0] = splits.split[j];
                }
                else {
                    wrapped->nb[2] = wrapped->nb[2] / wrapped->ne[1] * splits.split[j];
                    wrapped->nb[3] = wrapped->nb[3] / wrapped->ne[1] * splits.split[j];

                    // update row count
                    wrapped->ne[1] = splits.split[j];
                }
            }
        }
        else {
            if (ctx->split) {
                GGML_LOG_ERROR("ggml_backend_tp_buffer_init_tensor: split buffer type %s does not support views\n", buffer->buft->iface.get_name(buffer->buft));
                return GGML_STATUS_FAILED;
            }
            auto view_src_extra = (ggml_tensor_parallel_extra *)tensor->view_src->extra;
            auto view_src = view_src_extra->tensors[j];
            if (!tensor_is_split_compatible && view_src_extra->split_tensors) {
                ensure_rejoined(tensor->view_src);
                view_src = view_src_extra->rejoined_tensors[j];
            }
            if (tensor->view_offs % alignment) {
                GGML_LOG_ERROR("ggml_backend_tp_buffer_init_tensor: view_offs %zu is not a multiple of parallel alignment %zu\n", tensor->view_offs, device_alignment);
                return GGML_STATUS_FAILED;
            }
            // the view on this tensor must be adjusted to the device's alignment for heterogenous devices that may have different alignments
            auto view_offs = tensor->view_offs / alignment * device_alignment;
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
            auto backend_buffer = ctx->backend_buffers[j];
            auto wrapped = extra->tensors[j];
            auto split_offset = cur_row * bytes_per_row;

            // the split tensors should have the same alignment as the wrapping tensor, and thus the same stride.
            if (wrapped->nb[1] != tensor->nb[1]) {
                GGML_LOG_ERROR("ggml_backend_tp_buffer_set_tensor: wrapped->nb[1] %zu != tensor->nb[1] %zu\n", wrapped->nb[1], tensor->nb[1]);
                return;
            }

            auto split_size = (size_t) splits.split[j] * bytes_per_row;
            backend_buffer->iface.set_tensor(backend_buffer, wrapped, (const char *) data + split_offset, 0, split_size);

            cur_row += splits.split[j];
        }

        extra->original_data = (char *)malloc(size);
        memcpy(extra->original_data, data, size);
        extra->original_size = size;
    }
    else {
        for (size_t j = 0; j < ggml_parallel_devices.size(); j++) {
            auto wrapped = extra->tensors[j];
            auto backend_buffer = ctx->backend_buffers[j];
            backend_buffer->iface.set_tensor(backend_buffer, wrapped, data, offset, size);
        }
    }
}

static void ggml_backend_tp_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_tensor_parallel_extra * extra = (ggml_tensor_parallel_extra *)tensor->extra;

    ensure_rejoined(tensor);
    rejoin_tensor(tensor, extra, (char * )data);

    if (extra->split_tensors) {
        return;
        auto r = extra->rejoined_tensors[0];
        auto buft = r->buffer;
        // todo use async version
        buft->iface.get_tensor(buft, r, data, offset, size);
    }
    else {
        auto r = extra->tensors[0];
        auto buft = r->buffer;
        // todo use async version
        buft->iface.get_tensor(buft, r, data, offset, size);
    }


    for (auto be : ggml_parallel_backends) {
        ggml_backend_synchronize(be);
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

    ;
    max_alloc_size = ggml_align_size(max_alloc_size, ggml_backend_tp_buffer_type_get_alignment(buft));
    return max_alloc_size;
    // return ggml_nbytes(tensor);
}

static ggml_backend_buffer_type_i ggml_backend_tp_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_tp_buffer_type_name,
    /* .alloc_buffer     = */ ggml_backend_tp_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_tp_buffer_type_get_alignment,
    /* .get_max_size     = */ NULL,
    /* .get_alloc_size   = */ ggml_backend_tp_buffer_type_get_alloc_size,
    /* .is_host          = */ ggml_backend_cpu_buffer_type()->iface.is_host,
};

ggml_backend_buffer_type_t ggml_backend_tp_buffer_type(const char * endpoint) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    // NOTE: buffer types are allocated and never freed; this is by design
    static std::unordered_map<std::string, ggml_backend_buffer_type_t> buft_map;
    auto it = buft_map.find(endpoint);
    if (it != buft_map.end()) {
        return it->second;
    }
    // auto sock = get_socket(endpoint);
    // if (sock == nullptr) {
    //     fprintf(stderr, "Failed to connect to %s\n", endpoint);
    //     return nullptr;
    // }
    // size_t alignment = get_alignment(sock);
    // size_t max_size = get_max_size(sock);
    ggml_backend_tp_buffer_type_context * buft_ctx = new ggml_backend_tp_buffer_type_context {
        /* .split = */ false,
    };

    ggml_backend_buffer_type_t buft = new ggml_backend_buffer_type {
        /* .iface   = */ ggml_backend_tp_buffer_type_interface,
        /* .device  = */ ggml_backend_tp_reg_get_device(ggml_backend_tp_reg(), 0),
        /* .context = */ buft_ctx
    };
    buft_map[endpoint] = buft;
    return buft;
}

static ggml_backend_t ggml_backend_tp_device_init(ggml_backend_dev_t dev, const char * params) {
    ggml_backend_tp_device_context * ctx = (ggml_backend_tp_device_context *)dev->context;

    return ggml_backend_tp_init(ctx->endpoint.c_str());

    GGML_UNUSED(params);
}

static ggml_backend_buffer_type_t ggml_backend_tp_device_get_buffer_type(ggml_backend_dev_t dev) {
    ggml_backend_tp_device_context * ctx = (ggml_backend_tp_device_context *)dev->context;

    return ggml_backend_tp_buffer_type(ctx->endpoint.c_str());

    GGML_UNUSED(dev);
}

static bool ggml_backend_tp_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    GGML_UNUSED(dev);
    GGML_UNUSED(op);

    if (op->op == GGML_OP_RESHAPE) {
        printf("ggml_backend_tp_device_supports_op: op %s is not supported\n", ggml_op_name(op->op));
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

#define NUM_DEVICES 1

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
            printf("ggml_backend_tp_reg: registered device %s (%s)\n",
                ggml_backend_dev_name(device), ggml_backend_dev_description(device));

            ggml_parallel_devices.push_back(device);
        }
    }
    return true;
}

static ggml_backend_buffer_type_i ggml_backend_tp_split_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_tp_split_buffer_type_name,
    /* .alloc_buffer     = */ ggml_backend_tp_buffer_type_interface.alloc_buffer,
    /* .get_alignment    = */ ggml_backend_tp_buffer_type_interface.get_alignment,
    /* .get_max_size     = */ NULL,//ggml_backend_tp_get_max_size,
    /* .get_alloc_size   = */ ggml_backend_tp_buffer_type_interface.get_alloc_size,
    /* .is_host          = */ ggml_backend_tp_buffer_type_interface.is_host,
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
