#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml.h>

#ifdef GGML_USE_CUBLAS
#include <ggml-cuda.h>
#endif

#ifdef GGML_USE_METAL
#include <ggml-metal.h>
#endif

#ifdef GGML_USE_VULKAN
#include <ggml-vulkan.h>
#endif

#include <cmath>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

struct magnet_hparams {
    int32_t dim = 1024;
    int32_t num_heads = 16;
    int32_t num_layers = 24;
    int32_t hidden_scale = 4;
    int32_t n_q = 4;
    int32_t kv_repeat = 1;
    int32_t card = 2048;
    int32_t subcodes_context = 5;
    int32_t sample_rate = 32000;
};

struct magnet_transformer_block {
    // Self MHA
    // The q, k, v matricies are derived from this tensor
    struct ggml_tensor* self_attn_in_proj_w;
    // nn.Linear applied to output of attention
    struct ggml_tensor* self_attn_out_proj_w;

    // nn.Linear
    struct ggml_tensor* linear1_w;
    struct ggml_tensor* linear2_w;

    // nn.LayerNorm
    // elementwise_affine=True
    struct ggml_tensor* layer_norm1_w;
    struct ggml_tensor* layer_norm1_b;

    // nn.LayerNorm
    // elementwise_affine=True
    struct ggml_tensor* layer_norm2_w;
    struct ggml_tensor* layer_norm2_b;

    // Cross MHA
    struct ggml_tensor* cross_attn_in_proj_w;
    struct ggml_tensor* cross_attn_out_proj_w;

    // nn.LayerNorm
    // elementwise_affine=True
    struct ggml_tensor* norm_cross_w;
    struct ggml_tensor* norm_cross_b;
};

struct magnet_transformer {
    std::vector<magnet_transformer_block> transformer_blocks;
};

struct magnet_model {
    // See audiocraft T5Conditioner
    struct ggml_tensor* conditioning_w;
    struct ggml_tensor* conditioning_b;

    // nn.Embedding
    // Scaled embedding for n_q codebooks
    struct ggml_tensor* embed0_w;
    struct ggml_tensor* embed1_w;
    struct ggml_tensor* embed2_w;
    struct ggml_tensor* embed3_w;

    magnet_transformer transformer;

    // nn.LayerNorm
    struct ggml_tensor* out_norm_w;
    struct ggml_tensor* out_norm_b;

    // nn.Linear w/o bias for n_q codebooks
    struct ggml_tensor* linear0_w;
    struct ggml_tensor* linear1_w;
    struct ggml_tensor* linear2_w;
    struct ggml_tensor* linear3_w;

    magnet_hparams hparams;
    struct ggml_context* ctx;
    ggml_backend_t backend;
    struct ggml_backend_buffer* buffer;
};

struct magnet_context {
    magnet_model model;

    struct ggml_gallocr* galloc;
    ggml_tallocr talloc;
};

static void ggml_log_callback_default(ggml_log_level level, const char* text, void* user_data)
{
    (void)level;
    (void)user_data;
    fputs(text, stderr);
    fflush(stderr);
}

#define MAGNET_INFILE_MAGIC 0x46554747 // 'GGUF' LE
#define GGUF_GET_I32(ctx, key) gguf_get_val_i32(ctx, gguf_find_key(ctx, key))

// FIXME: remove this
#define MAX_ELEMENTS_PER_DIM 100
#define MIN(a, b) ((a) < (b) ? (a) : (b))

static float get_value(const struct ggml_tensor* tensor, int64_t idx)
{
    if (tensor->type == GGML_TYPE_F32) {
        return ((float*)tensor->data)[idx];
    } else { // Assuming F16
        return ggml_fp16_to_fp32(((ggml_fp16_t*)tensor->data)[idx]);
    }
}

static void print_value(float value)
{
    if (std::isnan(value))
        printf("nan");
    else if (std::isinf(value))
        printf("%sinf", value > 0 ? "" : "-");
    else
        printf("%.4f", value);
}

static void print_dim(const struct ggml_tensor* tensor, int64_t offset, int64_t stride, int64_t size, int indent)
{
    printf("%*s[", indent * 2, "");
    int64_t print_count = MIN(size, MAX_ELEMENTS_PER_DIM);

    for (int64_t i = 0; i < print_count; i++) {
        if (i == MAX_ELEMENTS_PER_DIM / 2 && size > MAX_ELEMENTS_PER_DIM) {
            printf("..., ");
            i = size - MAX_ELEMENTS_PER_DIM / 2 - 1;
        } else {
            print_value(get_value(tensor, offset + i * stride));
            if (i < print_count - 1)
                printf(", ");
        }
    }
    printf("]");
}

void print_tensor(const struct ggml_tensor* tensor)
{
    if (!tensor || !tensor->data) {
        printf("Tensor or tensor data is NULL\n");
        return;
    }

    int n_dims = ggml_n_dims(tensor);
    printf("Tensor dimensions: ");
    for (int i = 0; i < n_dims; i++) {
        printf("%lld", tensor->ne[i]);
        if (i < n_dims - 1)
            printf(" x ");
    }
    printf("\nTensor type: %s\n", ggml_type_name(tensor->type));

    if (tensor->type != GGML_TYPE_F16 && tensor->type != GGML_TYPE_F32) {
        printf("Warning: This function only prints F16 and F32 tensors correctly.\n");
        return;
    }

    printf("tensor(");
    int64_t offset = 0;
    int64_t stride = 1;
    for (int d = 0; d < n_dims; d++) {
        print_dim(tensor, offset, stride, tensor->ne[d], n_dims - d - 1);
        if (d < n_dims - 1) {
            printf(",\n");
            stride *= tensor->ne[d];
        }
    }
    printf(")\n");
}

bool load_parameters(std::string& file_name, magnet_model& model)
{
    // Load le model
    {
        struct ggml_init_params params = {
            .mem_size = 0,
            .mem_buffer = NULL,
        };
        model.ctx = ggml_init(params);

        // Now try to init from the file
        struct gguf_init_params gguf_params {
            .no_alloc = false,
            .ctx = &model.ctx,
        };

        struct gguf_context* gguf_ctx = gguf_init_from_file(file_name.c_str(), gguf_params);
        if (gguf_ctx == nullptr) {
            fprintf(stderr, "%s: Failed to load gguf file\n", __func__);
            return false;
        }

        model.hparams.dim = GGUF_GET_I32(gguf_ctx, "params.dim");
        model.hparams.num_heads = GGUF_GET_I32(gguf_ctx, "params.num_heads");
        model.hparams.num_layers = GGUF_GET_I32(gguf_ctx, "params.num_layers");
        model.hparams.hidden_scale = GGUF_GET_I32(gguf_ctx, "params.hidden_scale");
        model.hparams.n_q = GGUF_GET_I32(gguf_ctx, "params.n_q");
        model.hparams.kv_repeat = GGUF_GET_I32(gguf_ctx, "params.kv_repeat");
        model.hparams.card = GGUF_GET_I32(gguf_ctx, "params.card");
        model.hparams.subcodes_context = GGUF_GET_I32(gguf_ctx, "params.subcodes_context");
        model.hparams.sample_rate = GGUF_GET_I32(gguf_ctx, "params.sample_rate");

        printf("Model Hyperparameters\n");
        printf("dim:                %d\n", model.hparams.dim);
        printf("num_heads:          %d\n", model.hparams.num_heads);
        printf("num_layers:         %d\n", model.hparams.num_layers);
        printf("n_q:                %d\n", model.hparams.n_q);
        printf("card:               %d\n", model.hparams.card);
        printf("hidden_scale:       %d\n", model.hparams.hidden_scale);
        printf("kv_repeat:          %d\n", model.hparams.kv_repeat);
        printf("subcodes_context:   %d\n", model.hparams.subcodes_context);
        printf("sample_rate:        %d\n", model.hparams.sample_rate);

        int n_keys = gguf_get_n_kv(gguf_ctx);
        printf("Number of keys: %d\n", n_keys);
        int n_tensors = gguf_get_n_tensors(gguf_ctx);
        printf("Number of tensors: %d\n", n_tensors);

#ifdef GGML_USE_VULKAN
        ggml_vk_instance_init()
            model.backend
            = ggml_backend_vk_init(0);
#endif

        if (!model.backend) {
            model.backend = ggml_backend_cpu_init();
        }

        gguf_free(gguf_ctx);
    }

    {
        auto& hparams = model.hparams;
        ggml_context* ctx = model.ctx;
        auto n_q = hparams.n_q;

        GGML_ASSERT(ggml_get_first_tensor(ctx) != nullptr);

        // Embeddings
        // NOTE: can support more than n_q codebooks, however both small & medium models only use 4 so hardcoded
        model.embed0_w = ggml_get_tensor(ctx, "emb.0.weight");
        model.embed1_w = ggml_get_tensor(ctx, "emb.1.weight");
        model.embed2_w = ggml_get_tensor(ctx, "emb.2.weight");
        model.embed3_w = ggml_get_tensor(ctx, "emb.3.weight");
        printf("Embedding weight shape (%d, %d)\n", model.embed0_w->ne[0], model.embed0_w->ne[1]);

        // Linear Layers
        model.linear0_w = ggml_get_tensor(ctx, "linears.0.weight");
        model.linear1_w = ggml_get_tensor(ctx, "linears.1.weight");
        model.linear2_w = ggml_get_tensor(ctx, "linears.2.weight");
        model.linear3_w = ggml_get_tensor(ctx, "linears.3.weight");
        printf("Linear weight shape (%d, %d)\n", model.linear0_w->ne[0], model.linear0_w->ne[1]);

        // Normalization
        model.out_norm_w = ggml_get_tensor(ctx, "out_norm.weight");
        model.out_norm_b = ggml_get_tensor(ctx, "out_norm.bias");
        printf("out_norm weight & bias shape (%d) (%d)\n", model.out_norm_w->ne[0], model.out_norm_b->ne[0]);

        model.transformer = magnet_transformer();
        auto& transformer = model.transformer;

        // Reserve num_layers transformer blocks
        transformer.transformer_blocks.resize(hparams.num_layers);
        for (int i = 0; i < hparams.num_layers; i++) {
            auto& layer = transformer.transformer_blocks[i];
            char tmp_name[255];

#define CHECK_SHAPE(tensor) \
    GGML_ASSERT(tmp_name);  \
    // printf("%s shape: (%d, %d)\n", tmp_name, tensor->ne[0], tensor->ne[1]);

            // Under the assumption that the layers are contiguous, save some time from lookup
            snprintf(tmp_name, 255, "transformer.layers.%d.self_attn.in_proj_weight", i);
            layer.self_attn_in_proj_w = ggml_get_tensor(ctx, tmp_name);
            CHECK_SHAPE(layer.self_attn_in_proj_w);

            snprintf(tmp_name, 255, "transformer.layers.%d.self_attn.out_proj.weight", i);
            layer.self_attn_out_proj_w = ggml_get_next_tensor(ctx, layer.self_attn_in_proj_w);
            CHECK_SHAPE(layer.self_attn_out_proj_w);

            snprintf(tmp_name, 255, "transformer.layers.%d.linear1", i);
            layer.linear1_w = ggml_get_next_tensor(ctx, layer.self_attn_out_proj_w);
            CHECK_SHAPE(layer.linear1_w);

            snprintf(tmp_name, 255, "transformer.layers.%d.linear2", i);
            layer.linear2_w = ggml_get_next_tensor(ctx, layer.linear1_w);
            CHECK_SHAPE(layer.linear2_w);

            snprintf(tmp_name, 255, "transformer.layers.%d.norm1.weight", i);
            layer.layer_norm1_w = ggml_get_next_tensor(ctx, layer.linear2_w);
            CHECK_SHAPE(layer.layer_norm1_w);

            snprintf(tmp_name, 255, "transformer.layers.%d.norm1.bias", i);
            layer.layer_norm1_b = ggml_get_next_tensor(ctx, layer.layer_norm1_w);
            CHECK_SHAPE(layer.layer_norm1_b);

            snprintf(tmp_name, 255, "transformer.layers.%d.norm2.weight", i);
            layer.layer_norm2_w = ggml_get_next_tensor(ctx, layer.layer_norm1_b);
            CHECK_SHAPE(layer.layer_norm2_w);

            snprintf(tmp_name, 255, "transformer.layers.%d.norm2.bias", i);
            layer.layer_norm2_b = ggml_get_next_tensor(ctx, layer.layer_norm2_w);
            CHECK_SHAPE(layer.layer_norm2_b);

            snprintf(tmp_name, 255, "transformer.layers.%d.cross_attention.in_proj_weight", i);
            layer.cross_attn_in_proj_w = ggml_get_next_tensor(ctx, layer.layer_norm2_b);
            CHECK_SHAPE(layer.cross_attn_in_proj_w);

            snprintf(tmp_name, 255, "transformer.layers.%d.cross_attention.out_proj.weight", i);
            layer.cross_attn_out_proj_w = ggml_get_next_tensor(ctx, layer.cross_attn_in_proj_w);
            CHECK_SHAPE(layer.cross_attn_out_proj_w);

            snprintf(tmp_name, 255, "transformer.layers.%d.norm_cross.weight", i);
            layer.norm_cross_w = ggml_get_next_tensor(ctx, layer.cross_attn_out_proj_w);
            CHECK_SHAPE(layer.norm_cross_w);

            snprintf(tmp_name, 255, "transformer.layers.%d.norm_cross.bias", i);
            layer.norm_cross_b = ggml_get_next_tensor(ctx, layer.norm_cross_w);
            CHECK_SHAPE(layer.norm_cross_b);
        }
    }

    return true;
}

#define PRINT_SHAPE(tensor) printf("(%d, %d, %d, %d)\n", tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);

// 2.1) Normalize (LayerNorm https://arxiv.org/pdf/1607.06450)
ggml_tensor* layer_norm_forward(ggml_context* ctx, ggml_tensor* w, ggml_tensor* b, ggml_tensor* x)
{
    // layer_norm = ((x - mean) / sqrt(variance(x))) * weight + bias
    // Saw from codebase that ggml_norm does the variance & mean calculations already!!
    return ggml_add_inplace(ctx, ggml_mul_inplace(ctx, ggml_norm(ctx, x, 1e-5f), w), b);
}

// Multi-headed attention (standard)
// In the future, use Flash Attention algorithm. https://arxiv.org/pdf/2205.14135
ggml_tensor* multihead_attn_forward(ggml_context* ctx, ggml_tensor* k, ggml_tensor* q, ggml_tensor* v, ggml_tensor* x, int32_t num_heads, bool cross_attn)
{
    return nullptr;
}

// Linear transformation layer
ggml_tensor* magnet_linear_forward(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b = nullptr)
{
    ggml_tensor* out = ggml_mul_mat(ctx, x, w);
    if (b != nullptr) {
        out = ggml_add(ctx, out, b);
    }
    return out;
}

ggml_tensor* magnet_transformer_block_forward(magnet_model* model, ggml_context* ctx, magnet_transformer_block* block, ggml_tensor* x)
{
    const auto& hparams = model->hparams;
    PRINT_SHAPE(block->layer_norm1_w);
    // 2.1) Normalize (LayerNorm https://arxiv.org/pdf/1607.06450)
    x = layer_norm_forward(ctx, ggml_repeat(ctx, block->layer_norm1_w, x), ggml_repeat(ctx, block->layer_norm1_b, x), x);
    printf("After first layernorm: ");
    PRINT_SHAPE(x);

    // 2.2) Self attention (use Flash Attention, see paper https://arxiv.org/abs/2205.14135)

    // Forward Linear of projected weights teehee
    struct ggml_tensor* projected = magnet_linear_forward(ctx, x, block->self_attn_in_proj_w);
    printf("After forward pass: ");
    PRINT_SHAPE(projected);

    auto dim = hparams.dim;
    auto card = hparams.card;
    auto hidden_scale = hparams.hidden_scale;
    auto embed_dim = card + 1;
    auto start = embed_dim;
    auto per_head_dim = dim / hparams.num_heads;
    auto kv_heads = hparams.num_heads / hparams.kv_repeat;
    auto end = (start + per_head_dim * kv_heads) - 1;

    printf("per_head_dim: %d, kv_heads: %d, kv_repeat: %d, embed_dim: %d, start: %d, end: %d\n", per_head_dim, kv_heads, hparams.kv_repeat, embed_dim, start, end);
    // EXPECTING THE LAYOUT (b, t, h, d)

    struct ggml_tensor* k = ggml_view_3d(ctx, projected, projected->ne[0], projected->ne[1], embed_dim, projected->nb[1], projected->nb[2], 0);
    printf("k :");
    PRINT_SHAPE(k);
    struct ggml_tensor* q = ggml_view_3d(ctx, projected, projected->ne[0], projected->ne[1], start - end, projected->nb[1], start * ggml_type_size(projected->type), 0);
    struct ggml_tensor* v = ggml_view_3d(ctx, projected, projected->ne[0], projected->ne[1], projected->ne[2] - end, projected->nb[1], end * ggml_type_size(projected->type), 0);

    printf("q :");
    PRINT_SHAPE(q);
    printf("v :");
    PRINT_SHAPE(v);

    // 2.3) Cross attn normalization (LayerNorm). This is done with the provided conditions
    // 2.4) Cross attention
    // 2.5) Feedforward block (linears)
    // 2.6) Normalize (LayerNorm)

    return projected;
}

// Positional encoding must be a custom operation
void magnet_positional_encoding(struct ggml_tensor* dst, const struct ggml_tensor* src, int ith, int nth, void* userdata)
{
    GGML_ASSERT(ggml_are_same_shape(dst, src));
    // at the moment only have 2 dimensions so don't care about other channels!
    const auto MAX_PERIOD = 10000;
    for (int i = 0; i < src->ne[0]; i++) {
        for (int pos = 0; pos < src->ne[i]; pos++) {
            float inner = pos / pow(MAX_PERIOD, (2.0 * i) / 1024.0);
            float val = (i % 2 == 0) ? cos(inner) : sin(inner);
            ggml_set_f32_nd(dst, i, pos, 0, 0, val);
        }
    }
}

ggml_tensor* magnet_transformer_forward(magnet_model* model, ggml_context* ctx, ggml_tallocr* alloc, ggml_tensor* x)
{
    // Expected shape (B, K, S)
    auto batch_size = x->ne[0];
    auto tokens = x->ne[1];
    auto channels = x->ne[2];

    // 1) Create position embeddings (MAGNeT uses sine embeddings)
    ggml_tensor* positions = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, batch_size, tokens, channels);
    ggml_tallocr_alloc(alloc, positions);

    // Positional encoding!!!! :D
    positions = ggml_map_custom1(ctx, positions, magnet_positional_encoding, GGML_N_TASKS_MAX, NULL);
    x = ggml_add(ctx, positions, x);

    // 2) Apply each transformer Layer
    auto& blocks = model->transformer.transformer_blocks;
    for (int i = 0; i < blocks.size(); i++) {
        x = magnet_transformer_block_forward(model, ctx, &blocks[i], x);
    }

    return x;
}

ggml_cgraph* build_graph(magnet_model& model, struct ggml_tallocr* allocr)
{
    static size_t buf_size = ggml_tensor_overhead() * 100000 + ggml_graph_overhead() + (1024 * 1024 * 1024);
    static std::vector<uint8_t> buf(buf_size);

    // create dummy context
    struct ggml_init_params params0 = {
        .mem_size = buf_size,
        .mem_buffer = buf.data(),
        .no_alloc = true
    };

    // create temporary context to build the grpah
    struct ggml_context* ctx0 = ggml_init(params0);

    struct ggml_cgraph* gf = ggml_new_graph(ctx0);

    struct ggml_tensor* input = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 1, 1024, 2);
    ggml_tallocr_alloc(allocr, input);

    // apply the embeddings before passing it to the transformer

    srand(time(NULL));
    for (int c = 0; c < input->ne[2]; c++) {
        for (int i = 0; i < input->ne[1]; i++) {
            ggml_set_f32_nd(input, 1, i, c, 1, rand());
        }
    }
    PRINT_SHAPE(input);

    // FIXME: implement other modules, starting with layernorm to make sure stuff works
    // struct ggml_tensor* result = layer_norm_forward(ctx0, w, b, input);
    struct ggml_tensor* result = magnet_transformer_forward(&model, ctx0, allocr, input);

    ggml_build_forward_expand(gf, result);

    ggml_free(ctx0);
    return gf;
}

int main(int argc, char** argv)
{
    magnet_context* magnet_ctx = new magnet_context();
    magnet_ctx->model = magnet_model();

    std::string file_name = "C:\\Users\\drew\\project\\magnet.cpp\\mdl\\small\\ggml_model.bin";
    if (!load_parameters(file_name, magnet_ctx->model)) {
        fprintf(stderr, "%s: Failed to load model parameters\n", __func__);
        return -1;
    }

    auto& ctx = magnet_ctx->model.ctx;
    GGML_ASSERT(ctx != nullptr);
    auto& model = magnet_ctx->model;

    magnet_ctx->galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
    model.buffer = ggml_backend_alloc_buffer(model.backend, 1024 * 1024 * 1024); // fuck it, give it a gig!
    magnet_ctx->talloc = ggml_tallocr_new(model.buffer);

    ggml_cgraph* graph = build_graph(model, &magnet_ctx->talloc);
    ggml_gallocr_alloc_graph(magnet_ctx->galloc, graph);
    ggml_graph_print(graph);

    ggml_backend_graph_compute(model.backend, graph);
    auto out = graph->nodes[graph->n_nodes - 1];
    print_tensor(out);

    ggml_free(magnet_ctx->model.ctx);
    ggml_backend_free(model.backend);
    ggml_gallocr_free(magnet_ctx->galloc);
    delete magnet_ctx;
    return 0;
}