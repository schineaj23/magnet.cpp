#include <stdio.h>
#include <vector>
#include <fstream>
#include "ggml.h"

struct magnet_hparams {
    int32_t dim = 1024;
    int32_t num_heads = 16;
    int32_t num_layers = 24;
    int32_t hidden_scale = 4;
    int32_t n_q = 4;
    int32_t kv_repeat = 1;
    int32_t card = 2048;
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

template<typename T>
static void read_safe(std::ifstream &infile, T &dest) {
    infile.read((char*)&dest, sizeof(T));
}

static void ggml_log_callback_default(ggml_log_level level, const char *text, void *user_data) {
    (void)level;
    (void)user_data;
    fputs(text, stderr);
    fflush(stderr);
}

#define MAGNET_INFILE_MAGIC 0x46554747 // 'GGUF' LE

bool load_parameters(std::string& file_name, magnet_model &model) {
    // Calculate the size in memory of the tensors for the context
    // Guess that this is useless now that I'm just using the builtin GGUF functions
    int ctx_size = 0;
    {
        auto& hparams = model.hparams;


        // TODO: read in the hyperparams from the GGUF file
        auto n_q = hparams.n_q;
        auto input_dim = hparams.dim;
        auto num_heads = hparams.num_heads;
        auto num_layers = hparams.num_layers;
        auto kv_repeat = hparams.kv_repeat;
        auto card = hparams.card;
        auto hidden_scale = hparams.hidden_scale;

        auto embed_dim = card + 1;
        auto dim_feedforward = hidden_scale * input_dim;

        // Conditioner (HF MAGNeT checkpoints use T5)
        // The parameters here depend on the model used
        // In the case of magnet-small-30secs, it uses t5-base
        // t5-base output dimension is 768- hardcoding this for now
        // FIXME: read the conditioning parameters from the hparams file
        auto conditioning_dim = 768;
        // The T5 model used in this case has an nn.Linear (conditioning_dim, input_dim)
        ctx_size += conditioning_dim * input_dim * ggml_type_size(GGML_TYPE_F32); // weight
        ctx_size += input_dim * ggml_type_size(GGML_TYPE_F32); // bias

        // Linear layers for each codebook (input_dim, card)
        for(int i = 0;i<n_q;i++) {
            ctx_size += input_dim * card * ggml_type_size(GGML_TYPE_F32);
        }

        // Embeddings (nn.Linear)
        for(int i=0;i<n_q;i++) {
            // emb0-4 (embed_dim, input_dim)
            ctx_size += embed_dim * input_dim * ggml_type_size(GGML_TYPE_F32);
        }

        // out_norm (nn.LayerNorm) weight & bias = input_dim
        ctx_size += 2 * input_dim * ggml_type_size(GGML_TYPE_F32);

        // Transformer Block
        for(int i=0;i<num_layers;i++) {
            // First Linear layer (1024, 4096)
            ctx_size += input_dim * dim_feedforward * ggml_type_size(GGML_TYPE_F16);

            // Second Linear Layer (4096, 1024)
            ctx_size += dim_feedforward * input_dim * ggml_type_size(GGML_TYPE_F16);

            // Normalization weight & bias (equivalent to nn.LayerNorm)
            // norm1 (1024)
            ctx_size += 2 * input_dim * ggml_type_size(GGML_TYPE_F16);
            // norm2 (1024)
            ctx_size += 2 * input_dim * ggml_type_size(GGML_TYPE_F16);
            // norm_cross (1024)
            ctx_size += 2 * input_dim * ggml_type_size(GGML_TYPE_F16);

            // self_attn (MHA) input_proj is qkv weights
            auto out_dim = input_dim;
            auto num_kv = num_heads / kv_repeat;
            auto kv_dim = (input_dim / num_heads) * num_kv;
            out_dim += 2 * kv_dim;
            // in_proj_weight (implemented as nn.Linear in AC) (embed_dim, out_dim)
            ctx_size += out_dim * input_dim * ggml_type_size(GGML_TYPE_F16);
            // out_proj_weight (1024, 1024)
            ctx_size += input_dim * input_dim * ggml_type_size(GGML_TYPE_F16);

            // cross_attention (MHA), follows exact same as self_attn
            // in_proj_weight (implemented as nn.Linear in AC) (embed_dim, out_dim)
            ctx_size += out_dim * input_dim * ggml_type_size(GGML_TYPE_F16);
            // out_proj_weight (1024, 1024)
            ctx_size += input_dim * input_dim * ggml_type_size(GGML_TYPE_F16);
        }
        printf("Estimated size (MB): %6.2f\n", ctx_size / (1024.0 * 1024.0));
    }

    {
        struct ggml_init_params params = {
            .mem_size = ctx_size,
            .mem_buffer = NULL
        };

        struct ggml_context* ctx = ggml_init(params);
        if(ctx == nullptr) {
            fprintf(stderr, "%s: Failed to initialize ggml\n", __func__);
            return false;
        }
        model.ctx = ctx;

        // Now try to init from the file
        struct gguf_init_params gguf_params {
            .no_alloc = false,
            .ctx = &ctx,
        };

        struct gguf_context* gguf_ctx = gguf_init_from_file(file_name.c_str(), gguf_params);
        if(gguf_ctx == nullptr) {
            fprintf(stderr, "%s: Failed to load gguf file\n", __func__);
            return false;
        }
        int n_keys = gguf_get_n_kv(gguf_ctx);
        printf("num of keys: %d\n", n_keys);
        int n_tensors = gguf_get_n_tensors(gguf_ctx);
        printf("num of tensors: %d\n", n_tensors);

        gguf_free(gguf_ctx);

        struct ggml_tensor* tensor = ggml_get_first_tensor(ctx);
        while(tensor != nullptr) {
            int dims = ggml_n_dims(tensor);
            const char* name = ggml_get_name(tensor);
            const char* type_name = ggml_type_name(tensor->type);
            printf("%s (%d) (%s)\n", name, dims, type_name);

            tensor = ggml_get_next_tensor(ctx, tensor);
        }
    }

    return true;
}

int main(int argc, char** argv) {
    printf("Hello world!\n");

    magnet_context *ctx = new magnet_context();
    ctx->model = magnet_model();

    std::string file_name = "C:\\Users\\drew\\project\\magnet.cpp\\mdl\\ggml_model.bin";

    load_parameters(file_name, ctx->model);

    ggml_free(ctx->model.ctx);
    delete ctx;

    return 0;
}