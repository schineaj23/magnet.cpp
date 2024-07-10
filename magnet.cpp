#include "ggml.h"
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
};

struct magnet_context {
    magnet_model model;
};

static void ggml_log_callback_default(ggml_log_level level, const char* text, void* user_data)
{
    (void)level;
    (void)user_data;
    fputs(text, stderr);
    fflush(stderr);
}

// FIXME: evil hack to allocate more memory
struct ggml_context {
    size_t mem_size;
    void* mem_buffer;
    bool mem_buffer_owned;
    bool no_alloc;
    bool no_alloc_save; // this is used to save the no_alloc state when using scratch buffers

    int n_objects;

    struct ggml_object* objects_begin;
    struct ggml_object* objects_end;

    struct ggml_scratch scratch;
    struct ggml_scratch scratch_save;
};

#define MAGNET_INFILE_MAGIC 0x46554747 // 'GGUF' LE
#define GGUF_GET_I32(ctx, key) gguf_get_val_i32(ctx, gguf_find_key(ctx, key))

bool load_parameters(std::string& file_name, magnet_model& model)
{
    // Load le model
    {
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
        int n_keys = gguf_get_n_kv(gguf_ctx);
        printf("Number of keys: %d\n", n_keys);
        int n_tensors = gguf_get_n_tensors(gguf_ctx);
        printf("Number of tensors: %d\n", n_tensors);

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

// 2.1) Normalize (LayerNorm https://arxiv.org/pdf/1607.06450)
ggml_tensor* layer_norm_forward(ggml_context* ctx, ggml_tensor* w, ggml_tensor* b, ggml_tensor* x)
{
    // layernorm = ((weight / std deviation of layer) * (input - mean)) + bias
    ggml_tensor* mean = ggml_mean(ctx, x);
    ggml_tensor* error = ggml_sub(ctx, x, mean);
    ggml_tensor* variance = ggml_mean(ctx, ggml_sqr(ctx, error));
    ggml_tensor* std_dev = ggml_sqrt(ctx, variance);

    ggml_tensor* out = ggml_add(ctx, ggml_mul_mat(ctx, ggml_div(ctx, w, std_dev), error), b);
    return out;
}

// Multi-headed attention via the Flash Attention algorithm. https://arxiv.org/pdf/2205.14135
ggml_tensor* multihead_attn_forward(ggml_context* ctx, ggml_tensor* k, ggml_tensor* q, ggml_tensor* v, ggml_tensor* x, int32_t num_heads, bool cross_attn)
{
    return nullptr;
}

ggml_tensor* magnet_transformer_block_forward(magnet_model* model, magnet_transformer_block* block, ggml_tensor* x)
{
    auto& ctx = model->ctx;
    // 2.1) Normalize (LayerNorm https://arxiv.org/pdf/1607.06450)
    x = layer_norm_forward(ctx, block->layer_norm1_w, block->layer_norm1_b, x);
    // 2.2) Self attention (use Flash Attention, see paper https://arxiv.org/abs/2205.14135)
    // 2.3) Cross attn normalization (LayerNorm)
    // 2.4) Cross attention
    // 2.5) Feedforward block (linears)
    // 2.6) Normalize (LayerNorm)

    return x;
}

ggml_tensor* create_sin_embedding(magnet_model* model)
{
    return nullptr;
}

// FIXME: remove this
void print_tensor(struct ggml_tensor* tensor)
{
    if (tensor == NULL) {
        printf("Tensor is NULL\n");
        return;
    }

    int64_t ne0 = tensor->ne[0];
    int64_t ne1 = tensor->ne[1];
    int64_t ne2 = tensor->ne[2];
    int64_t ne3 = tensor->ne[3];

    printf("Tensor dimensions: %lld x %lld x %lld x %lld\n", ne0, ne1, ne2, ne3);
    printf("Tensor type: %d\n", tensor->type);

    float* data = (float*)tensor->data;

    for (int i3 = 0; i3 < ne3; i3++) {
        for (int i2 = 0; i2 < ne2; i2++) {
            for (int i1 = 0; i1 < ne1; i1++) {
                for (int i0 = 0; i0 < ne0; i0++) {
                    int64_t idx = i3 * ne2 * ne1 * ne0 + i2 * ne1 * ne0 + i1 * ne0 + i0;
                    printf("%f ", data[idx]);
                }
                printf("\n");
            }
            printf("\n");
        }
        printf("\n");
    }
}

ggml_tensor* magnet_transformer_forward(magnet_model* model, ggml_tensor* x)
{
    // Expected shape (B, T, C)
    auto batch_size = x->ne[0];
    auto tokens = x->ne[1];
    auto channels = x->ne[2];

    auto& ctx = model->ctx;
    // 1) Create position embeddings (MAGNeT uses sine embeddings)
    ggml_tensor* positions = ggml_arange(ctx, 0.0, tokens, 1);
    GGML_ASSERT(false);
    // since only want one element just take the type size?
    positions = ggml_view_3d(ctx, positions, 1, -1, 1, tokens * sizeof(tokens), sizeof(tokens), 0);
    // NOTE: audiocraft adds offsets since they are streaming their transformer but idgaf atm
    const auto MAX_PERIOD = 10000;
    auto half_dim = channels / 2;

    printf("Shape (%d, %d, %d)\n", positions->ne[0], positions->ne[1], positions->ne[2]);
    print_tensor(positions);

    for (int i = 0; i < channels; i++) {
        for (int t = 0; t < tokens; t++) {
            *(float*)((char*)positions->data + i * positions->nb[1] + t * positions->nb[2]) = 10.0;
        }
    }

    print_tensor(positions);

    // 2) Apply each transformer Layer
    // auto& blocks = model->transformer.transformer_blocks;
    // for (int i = 0; i < blocks.size(); i++) {
    //     x = magnet_transformer_block_forward(model, &blocks[i], x);
    // }

    return x;
}

int main(int argc, char** argv)
{
    magnet_context* magnet_ctx = new magnet_context();
    magnet_ctx->model = magnet_model();

    size_t mem_size = 1024 * 1024 * 1024;
    static void* buf = malloc(mem_size);
    struct ggml_init_params params = {
        .mem_size = mem_size,
        .mem_buffer = buf,
        .no_alloc = false,
    };

    magnet_ctx->model.ctx = ggml_init(params);
    auto& ctx = magnet_ctx->model.ctx;
    ggml_cgraph* gf = ggml_new_graph(ctx);
    GGML_ASSERT(gf);

    ggml_tensor* x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, 100, 2);
    ggml_set_param(ctx, x);

    printf("%f\n", ggml_used_mem(ctx) / (1024.0 * 1024.0));

    std::string file_name = "C:\\Users\\drew\\project\\magnet.cpp\\mdl\\small\\ggml_model.bin";
    if (!load_parameters(file_name, magnet_ctx->model)) {
        fprintf(stderr, "%s: Failed to load model parameters\n", __func__);
        return -1;
    }

    GGML_ASSERT(ctx != nullptr);
    printf("%f\n", ggml_used_mem(ctx) / (1024.0 * 1024.0));

    printf("starting forward\n");
    ggml_set_no_alloc(ctx, false);
    magnet_transformer_forward(&magnet_ctx->model, x);

    ggml_build_forward_expand(gf, x);
    ggml_graph_compute_with_ctx(ctx, gf, 8);

    // FIXME: remove
    system("pause");
    ggml_free(magnet_ctx->model.ctx);
    delete magnet_ctx;

    return 0;
}