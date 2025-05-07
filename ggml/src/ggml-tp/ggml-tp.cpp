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


static std::vector<ggml_backend_dev_t> ggml_parallel_devices;
static std::vector<ggml_backend_t> ggml_parallel_backends;

// TP data structures

static ggml_guid_t ggml_backend_tp_guid() {
    static ggml_guid guid = {0xa4, 0x1f, 0x3c, 0xd9, 0x87, 0x6b, 0x91, 0x22, 0xde, 0x33, 0xab, 0x7c, 0x58, 0x44, 0x9e, 0x01};
    return &guid;
}


struct ggml_backend_tp_context {
};

struct ggml_tensor_parallel_extra {
    ggml_tensor * tensor;
    int tp_candidate;
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

static ggml_tensor * unwrap_tensor(ggml_tensor * tensor, std::map<ggml_tensor *, ggml_tensor *> & tensor_map) {
    auto found = tensor_map.find(tensor);
    if (found != tensor_map.end()) {
        return found->second;
    }
    ggml_tensor_parallel_extra * extra = (ggml_tensor_parallel_extra *)tensor->extra;
    auto wrapped = extra->tensor;
    tensor_map[tensor] = wrapped;
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (tensor->src[i] != nullptr) {
            wrapped->src[i] = unwrap_tensor(tensor->src[i], tensor_map);
        }
        else {
            wrapped->src[i] = nullptr;
        }
    }
    return wrapped;
}

static void map_tp_candidates(ggml_tensor *tensor, std::vector<ggml_tensor *> & check) {
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (tensor->src[i] != nullptr) {
            auto src = tensor->src[i];
            if (!src) {
                continue;
            }
            auto src_extra = (ggml_tensor_parallel_extra *)src->extra;
            if (src_extra->tp_candidate) {
                continue;
            }

            switch (src->op) {
                // can handle some basic math ops
                case GGML_OP_ADD:
                case GGML_OP_SUB:
                case GGML_OP_MUL:
                case GGML_OP_DIV:
                // split loaded weights if they're needed for 
                case GGML_OP_NONE:
                // test
                case GGML_OP_RMS_NORM:
                    break;
                default:
                    // printf("skipping %s\n", ggml_op_name(src->op));
                    continue;
            }
            src_extra->tp_candidate = 1;
            check.push_back(src);
        }
    }
}

static enum ggml_status ggml_backend_tp_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    auto be = ggml_parallel_backends[0];

    std::vector<ggml_tensor *> check;
    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * tensor = cgraph->nodes[i];
        ggml_tensor_parallel_extra * extra = (ggml_tensor_parallel_extra *)tensor->extra;
        extra->tp_candidate = tensor->op == GGML_OP_MUL_MAT;
        check.push_back(tensor);
    }

    while (!check.empty()) {
        auto tensor = check.back();
        check.pop_back();
        map_tp_candidates(tensor, check);
    }

    std::map<ggml_tensor*, ggml_tensor*> tensor_map;
    std::vector<ggml_tensor*> graph_nodes;
    for (int i = 0; i < cgraph->n_nodes; i++) {
        graph_nodes.push_back(unwrap_tensor(cgraph->nodes[i], tensor_map));
    }
    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_status status = be->iface.node_compute(be, graph_nodes[i]);
        // printf("op %d: %s\n", i, ggml_op_name(graph_nodes[i]->op));
        if (status != GGML_STATUS_SUCCESS) {
            return status;
        }
    }
    ggml_backend_synchronize(be);
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
    uint64_t remote_ptr;
    ggml_backend_buffer_t wrapped;
    bool split = false;
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

static const char * ggml_backend_tp_split_buffer_type_name(ggml_backend_buffer_type_t buft) {
    return "TPSplit";
    GGML_UNUSED(buft);
}

static bool ggml_backend_buft_is_tp_split(ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name == ggml_backend_tp_split_buffer_type_name;
}

static uint64_t max_ptr = 0;

static enum ggml_status ggml_backend_tp_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    // if (tensor->view_src != NULL) {
    //     assert(tensor->view_src->buffer->buft == buffer->buft);
    //     return GGML_STATUS_SUCCESS;
    // }

    ggml_backend_tp_buffer_context * ctx = (ggml_backend_tp_buffer_context *)buffer->context;

    // struct ggml_init_params params {
    //     /*.mem_size   =*/ ggml_tensor_overhead(),
    //     /*.mem_buffer =*/ NULL,
    //     /*.no_alloc   =*/ true,
    // };
    // ggml_context_ptr ctx_ptr { ggml_init(params) };
    // GGML_ASSERT(ctx_ptr != nullptr);

    // ggml_tensor * result = ggml_new_tensor_4d(ctx_ptr.get(), (ggml_type) tensor->type,
    //     tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);

    ggml_tensor * wrapped = new ggml_tensor();
    ggml_set_name(wrapped, tensor->name);
    wrapped->type = (ggml_type) tensor->type;
    wrapped->flags = tensor->flags;
    wrapped->buffer = ctx->wrapped;

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

    if (!tensor->view_src) {
        auto base = (char *) ctx->wrapped->iface.get_base(ctx->wrapped);
        auto tensor_offset = (uint64_t) tensor->data - (uint64_t) 128;
        wrapped->data = base + tensor_offset;
        auto size = ctx->wrapped->buft->iface.get_alloc_size(ctx->wrapped->buft, tensor);
        if ((uint64_t)wrapped->data + size > max_ptr) {
            GGML_LOG_ERROR("ggml_backend_tp_buffer_init_tensor: tensor data pointer %p + size %zu exceeds max pointer %p\n",
                wrapped->data, size, (void *)max_ptr);
        }
    }
    else {
        ggml_tensor_parallel_extra * extra = (ggml_tensor_parallel_extra *)tensor->view_src->extra;
        auto view_src = extra->tensor;
        wrapped->data = (char *) view_src->data + tensor->view_offs;
        wrapped->view_src = view_src;
        wrapped->view_offs = tensor->view_offs;
        if (wrapped->view_src == NULL) {
            GGML_LOG_ERROR("ggml_backend_tp_buffer_init_tensor: view_src is NULL for tensor %s\n", tensor->name);
            return GGML_STATUS_FAILED;
        }
    }

    ggml_tensor_parallel_extra * extra = new ggml_tensor_parallel_extra {
        /* .tensor = */ wrapped,
        /* .tp_candidate = */ 0,
    };
    tensor->extra = extra;

    return ctx->wrapped->iface.init_tensor(ctx->wrapped, wrapped);
}

static void ggml_backend_tp_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_backend_tp_buffer_context * ctx = (ggml_backend_tp_buffer_context *)buffer->context;
    ggml_tensor_parallel_extra * extra = (ggml_tensor_parallel_extra *)tensor->extra;
    ctx->wrapped->iface.set_tensor(ctx->wrapped, extra->tensor, data, offset, size);
}

static void ggml_backend_tp_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_backend_tp_buffer_context * ctx = (ggml_backend_tp_buffer_context *)buffer->context;
    ggml_tensor_parallel_extra * extra = (ggml_tensor_parallel_extra *)tensor->extra;
    ctx->wrapped->iface.get_tensor(ctx->wrapped, extra->tensor, data, offset, size);

    GGML_UNUSED(ctx);
    GGML_UNUSED(tensor);
    GGML_UNUSED(data);
    GGML_UNUSED(offset);
    GGML_UNUSED(size);
}

static bool ggml_backend_tp_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    return true;
    GGML_UNUSED(buffer);
    GGML_UNUSED(src);
    GGML_UNUSED(dst);
}

static void ggml_backend_tp_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_tp_buffer_context * ctx = (ggml_backend_tp_buffer_context *)buffer->context;
    ctx->wrapped->iface.clear(ctx->wrapped, value);
    GGML_UNUSED(ctx);
    GGML_UNUSED(buffer);
    GGML_UNUSED(value);
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
    /* .reset           = */ NULL,
};

static ggml_backend_buffer_t ggml_backend_tp_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_tp_buffer_context * ctx = new ggml_backend_tp_buffer_context();
    ctx->split = ggml_backend_buft_is_tp_split(buft);
    auto buffer_type = ggml_parallel_devices[0]->iface.get_buffer_type(ggml_parallel_devices[0]);
    ctx->wrapped = buffer_type->iface.alloc_buffer(buffer_type, size);
    ctx->base_ptr = (void *) 128;
    max_ptr = (uint64_t)ctx->wrapped->iface.get_base(ctx->wrapped) + size;
    return ggml_backend_buffer_init(buft, ggml_backend_tp_buffer_interface, ctx, size);
}

static size_t ggml_backend_tp_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return ggml_parallel_devices[0]->iface.get_buffer_type(ggml_parallel_devices[0])->iface.get_alignment(ggml_parallel_devices[0]->iface.get_buffer_type(ggml_parallel_devices[0]));
    GGML_UNUSED(buft);
}

static size_t ggml_backend_tp_get_max_size(ggml_backend_buffer_type_t buft) {
    auto wrapped_buft = ggml_parallel_devices[0]->iface.get_buffer_type(ggml_parallel_devices[0])->iface;
    auto test = wrapped_buft.get_max_size(ggml_parallel_devices[0]->iface.get_buffer_type(ggml_parallel_devices[0]));
    return test;
    GGML_UNUSED(buft);
}

static size_t ggml_backend_tp_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    GGML_UNUSED(buft);
    GGML_UNUSED(tensor);

    auto wrapped_buft = ggml_parallel_devices[0]->iface.get_buffer_type(ggml_parallel_devices[0])->iface;
    return wrapped_buft.get_alloc_size(ggml_parallel_devices[0]->iface.get_buffer_type(ggml_parallel_devices[0]), tensor);

    // return ggml_nbytes(tensor);
}

static ggml_backend_buffer_type_i ggml_backend_tp_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_tp_buffer_type_name,
    /* .alloc_buffer     = */ ggml_backend_tp_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_tp_buffer_type_get_alignment,
    /* .get_max_size     = */ NULL,//ggml_backend_tp_get_max_size,
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

static bool is_split_compatible(ggml_op op) {
    switch (op) {
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

static bool ggml_backend_tp_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    GGML_UNUSED(dev);
    GGML_UNUSED(op);

    // split buffers can only be used with GGML_OP_MUL_MAT and some other basic ops
    if (!is_split_compatible(op->op)) {
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            if (op->src[i] && op->src[i]->buffer && ggml_backend_buft_is_tp_split(op->src[i]->buffer->buft)) {
                return false;
            }
        }
    }

    return ggml_parallel_devices[0]->iface.supports_op(ggml_parallel_devices[0], op);
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
