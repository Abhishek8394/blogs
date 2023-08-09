---
title: "llama.cpp explained"
date: 2023-07-22T01:20:00-07:00
draft: true
type: "page"
card_background: "#0288d1"
---
# The annotated llama.cpp

## The Setup

```c++
typedef void (*offload_func_t)(struct ggml_tensor * tensor);
```
Defines a type `offload_func_t`. So a variable of type `offload_func_t` , means pointer to a function that: 
  - Takes one parameter - a pointer to a tensor `(struct ggml_tensor *)`. 
  - And does not return anything.

To get an idea on why this is done, refer to [this excellent SO answer](https://stackoverflow.com/a/4295495). It greatly helps readability when you need pointers to functions.

Side note:

- `ggml_tensor` is like a tensor placeholder. It is what gets inserted in a graph, and whose value will be filled up during execution of graph.
- Offload functions set the tensor output backend to GPU. tensors are GPU-accelerated if any input or the output has been offloaded. 
If you need to use / return `offload_func_t` but don't have anything to do for the tensor, you may return the new op function already defined in llama.cpp.

```c++
/**
 * If you don't want gpu accelerated, you can provide `llama_nop`. Below is defintion of `llama_nop`. 
 */
void llama_nop(struct ggml_tensor * tensor) { // don't offload by default
    (void) tensor;
}
```
### ggml helpers
```c++
static void ggml_graph_compute_helper(std::vector<uint8_t> & buf, ggml_cgraph * graph, int n_threads) {
    struct ggml_cplan plan = ggml_graph_plan(graph, n_threads);

    if (plan.work_size > 0) {
        buf.resize(plan.work_size);
        plan.work_data = buf.data();
    }

    ggml_graph_compute(graph, &plan);
}
```
GGML requires you to allocate memory before executing a graph. 
- So first given a graph and number of threads, a plan is created of type `ggml_cplan`.
- Then, we resize our allocated memory in `buf` using `buf.resize`.
- Then we set `plan.work_data` to pointer to data held by `buf`.
- Finally, `ggml_graph_compute` will execute the graph.

Next up is the scratch buffer allotment, This looks like it has some magic numbers in it. Check the code below.

```c++
// used to store the compute graph tensors + non-scratch data
static const std::map<e_model, size_t> & MEM_REQ_EVAL()
{
    static std::map<e_model, size_t> k_sizes = {
        { MODEL_3B,   8ull * MB },
        { MODEL_7B,  10ull * MB },
        { MODEL_13B, 12ull * MB },
        { MODEL_30B, 16ull * MB },
        { MODEL_65B, 24ull * MB }, // guess
        { MODEL_70B, 24ull * MB },
    };
    return k_sizes;
}
```
The numbers above were derived by using the function `ggml_used_mem`. So we allocate more than enough buffer for a model first, then use `ggml_used_mem` function to get the actual memory used and update it above. Unless commented with `//guess`.

## The Layers

### LLamaLayer

Moving on, we will skip past the `struct llama_hparams`. This is a struct to hold model config. The next here is the `struct llama_layer`, again this just holds tensors as per the standard llama layer. It is very similar to the transformer layer, if you're not familiar with the transformer layer, check out this [excellent blog](https://jalammar.github.io/illustrated-transformer/).

This layer can be understood by following the huggingface llama decoder layer [LlamaDecoderLayer](https://github.com/huggingface/transformers/blob/641adca55832ed9c5648f54dcd8926d67d3511db/src/transformers/models/llama/modeling_llama.py#L376).

#### 1. attention
First up is the standard attention block, [hf impl](https://github.com/huggingface/transformers/blob/641adca55832ed9c5648f54dcd8926d67d3511db/src/transformers/models/llama/modeling_llama.py#L237):
- The query weight tensor `wq`.
- The key weight tensor `wk`.
- The value weight tensor `wv`.
- The output weight tensor `wo`. This layer transforms the attention output tensor to our `hidden_size` dimension.

#### 2. attention_norm
`attention_norm` ([hf impl](https://github.com/huggingface/transformers/blob/641adca55832ed9c5648f54dcd8926d67d3511db/src/transformers/models/llama/modeling_llama.py#L75)) is the weight for the `LlamaRMSNorm` layer. Why do we need weight for RMS norm (root mean square normalization)? To give weight to each dimension! 

Wait, neural net layers generally have a bias and a weight! Turns out, the original llama layer did not use a bias and hence we only define weight tensors here.

#### 3. ff
`ff` is the fully connected layer, in llama that involves 3 linear layers ([hf impl](https://github.com/huggingface/transformers/blob/641adca55832ed9c5648f54dcd8926d67d3511db/src/transformers/models/llama/modeling_llama.py#L191)).
- `w1`: The gate projection.
- `w2`: The up projection.
- `w3`: The down projection.


```c++
struct llama_layer {

    // attention
    struct ggml_tensor * wq;
    struct ggml_tensor * wk;
    struct ggml_tensor * wv;
    struct ggml_tensor * wo;

    // attention normalization
    struct ggml_tensor * attention_norm;

    // ff
    struct ggml_tensor * w1;
    struct ggml_tensor * w2;
    struct ggml_tensor * w3;

    // normalization
    struct ggml_tensor * ffn_norm;
};
```

### Vocab
Store vocab. Llama uses Byte Pair Encoding (BPE). In this scheme, each token has an associated score with it. Check out this [excellent post](https://huggingface.co/learn/nlp-course/chapter6/5?fw=pt) to learn more. In a nutshell, tokenize into individual characters, then merge the most frequent pair in your corpus. Rinse and repeat till you reach the desired vocab size.

```c++
struct llama_vocab {
    using id    = int32_t;
    using token = std::string;

    struct token_score {
        token tok;
        float score;
    };

    std::unordered_map<token, id> token_to_id;
    std::vector<token_score> id_to_token;
};
```

## The Llama Model

```c++
struct llama_model {
    /**
     * Enum denoting model variant (3B, 7B, etc.).
     */
    e_model type = MODEL_UNKNOWN;

    /**
     * Llama model hyper params (model config).
     */
    llama_hparams hparams;

    /**
     * Token embeddings. Converts token ids (int) to a tensor (list<float>)
     */
    struct ggml_tensor * tok_embeddings;
    /**
     * norm: TODO: Verify Hold the normalized output
     * output: TODO: Verify Hold the output.
     */
    struct ggml_tensor * norm;
    struct ggml_tensor * output;

    /**
     * List of llama layers.
     */
    std::vector<llama_layer> layers;

    /**
     * Number of llama layers to run on GPU.
     */
    int n_gpu_layers;

    // context
    struct ggml_context * ctx = NULL;

    // the model memory buffer
    llama_ctx_buffer buf;

    // model memory mapped file
    std::unique_ptr<llama_mmap> mapping;

    // objects representing data potentially being locked in memory
    llama_mlock mlock_buf;
    llama_mlock mlock_mmap;

    // for quantize-stats only
    std::vector<std::pair<std::string, struct ggml_tensor *>> tensors_by_name;

    int64_t t_load_us = 0;
    int64_t t_start_us = 0;

    llama_vocab vocab;

    ~llama_model() {
        if (ctx) {
            ggml_free(ctx);
        }

#ifdef GGML_USE_CUBLAS
        for (size_t i = 0; i < tensors_by_name.size(); ++i) {
            ggml_cuda_free_data(tensors_by_name[i].second);
        }
        ggml_cuda_free_scratch();
#elif defined(GGML_USE_CLBLAST)
        for (size_t i = 0; i < tensors_by_name.size(); ++i) {
            ggml_cl_free_data(tensors_by_name[i].second);
        }
#endif
    }
};

```

## The Llama Context

This class wraps a llama model and provides various functionalities such as:
- Measuring and storing duration times for various operations.
- Caching self attention.
- Manage scratch buffers, allocators, etc.

TODO: Add more details

```c++
struct llama_context {
    llama_context(const llama_model & model) : model(model), t_load_us(model.t_load_us), t_start_us(model.t_start_us) {}
    ~llama_context() {
        if (model_owner) {
            delete &model;
        }
#ifdef GGML_USE_METAL
        if (ctx_metal) {
            ggml_metal_free(ctx_metal);
        }
#endif
#ifdef LLAMA_USE_ALLOCATOR
        if (alloc) {
            ggml_allocr_free(alloc);
        }
#endif
    }

    std::mt19937 rng;

    bool has_evaluated_once = false;

    int64_t t_sample_us = 0;
    int64_t t_eval_us   = 0;
    int64_t t_p_eval_us = 0;

    int32_t n_sample = 0; // number of tokens sampled
    int32_t n_eval   = 0; // number of eval calls
    int32_t n_p_eval = 0; // number of tokens in eval calls for the prompt (with batch size > 1)

    const llama_model & model;

    bool model_owner = false;

    int64_t t_load_us;
    int64_t t_start_us;

    // key + value cache for the self attention
    struct llama_kv_cache kv_self;

    size_t mem_per_token = 0;

    // decode output (2-dimensional array: [n_tokens][n_vocab])
    std::vector<float> logits;
    bool logits_all = false;

    // input embedding (1-dimensional array: [n_embd])
    std::vector<float> embedding;

    // reusable buffer for `struct ggml_graph_plan.work_data`
    std::vector<uint8_t> work_buffer;

    // memory buffers used to evaluate the model
    // TODO: move in llama_state
    llama_ctx_buffer buf_compute;

#ifdef LLAMA_USE_ALLOCATOR
    llama_ctx_buffer buf_alloc;
    ggml_allocr * alloc = NULL;
#endif

#ifdef LLAMA_USE_SCRATCH
    llama_ctx_buffer buf_scratch[LLAMA_MAX_SCRATCH_BUFFERS];
    int    buf_last = 0;
    size_t buf_max_size[LLAMA_MAX_SCRATCH_BUFFERS] = { 0 };
#endif

#ifdef GGML_USE_METAL
    ggml_metal_context * ctx_metal = NULL;
#endif

#ifdef GGML_USE_MPI
    ggml_mpi_context * ctx_mpi = NULL;
#endif

    void use_buf(struct ggml_context * ctx, int i) {
#if defined(LLAMA_USE_SCRATCH)
        size_t last_size = 0;

        if (i == -1) {
            last_size = ggml_set_scratch(ctx, { 0, 0, nullptr, });
        } else {
            auto & buf = buf_scratch[i];
            last_size = ggml_set_scratch(ctx, { 0, buf.size, buf.addr, });
        }

        if (buf_last >= 0) {
            buf_max_size[buf_last] = std::max(buf_max_size[buf_last], last_size);
        }

        buf_last = i;
#else
        (void) i;
        (void) ctx;
#endif
    }

    size_t get_buf_max_mem(int i) const {
#if defined(LLAMA_USE_SCRATCH)
        return buf_max_size[i];
#else
        (void) i;
        return 0;
#endif
    }
};
```

## Optimizations
### The KV Cache (llama_kv_cache)
This struct will hold our key and value tensors in a cache. TODO: Come back to explain the members once we figure out how they are used.

```c++
struct llama_kv_cache {
    struct ggml_tensor * k = NULL;
    struct ggml_tensor * v = NULL;

    struct ggml_context * ctx = NULL;

    llama_ctx_buffer buf;

    int n; // number of tokens currently in the cache

    ~llama_kv_cache() {
        if (ctx) {
            ggml_free(ctx);
        }

#ifdef GGML_USE_CUBLAS
        ggml_cuda_free_data(k);
        ggml_cuda_free_data(v);
#endif // GGML_USE_CUBLAS
    }
};
```
