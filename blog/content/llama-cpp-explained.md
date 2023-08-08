---
title: "llama.cpp explained"
date: 2023-07-22T01:20:00-07:00
draft: true
type: "page"
---
# The annotated llama.cpp

## The Setup

```c++

/**
 * Re-define  `(*offload_func_t)(struct ggml_tensor * tensor)` to `void`. So everywhere you see `(*offload_func_t)(struct ggml_tensor * tensor)` , it means `void`. To get an idea on why this is done, refer to [this excellent SO answer](https://stackoverflow.com/a/4295495).
 * First `ggml_tensor` is like a tensor placeholder. It is what gets inserted in a graph, and whose value will be filled up during execution of graph.
 * Offload functions set the tensor output backend to GPU. tensors are GPU-accelerated if any input or the output has been offloaded.
 */
typedef void (*offload_func_t)(struct ggml_tensor * tensor);

/**
 * If you don't want gpu accelerated, you can provide `llama_nop`. Below is defintion of `llama_nop`. 
 */
void llama_nop(struct ggml_tensor * tensor) { // don't offload by default
    (void) tensor;
}

//
// ggml helpers
//

/**
 * GGML requires you to allocate memory before executing a graph. 
 * - So first given a graph and number of threads, a plan is created of type `ggml_cplan`.
 * - Then, we resize our allocated memory in `buf` using `buf.resize`.
 * - Then we set `plan.work_data` to pointer to data held by `buf`.
 * - Finally, `ggml_graph_compute` will execute the graph.
 */
static void ggml_graph_compute_helper(std::vector<uint8_t> & buf, ggml_cgraph * graph, int n_threads) {
    struct ggml_cplan plan = ggml_graph_plan(graph, n_threads);

    if (plan.work_size > 0) {
        buf.resize(plan.work_size);
        plan.work_data = buf.data();
    }

    ggml_graph_compute(graph, &plan);
}
```

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
Store vocab.

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
