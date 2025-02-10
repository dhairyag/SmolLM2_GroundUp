```bash
LlamaConfig {
  "_name_or_path": "HuggingFaceTB/SmolLM2-135M",
  "architectures": [
    "LlamaForCausalLM"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 0,
  "eos_token_id": 0,
  "head_dim": 64,
  "hidden_act": "silu",
  "hidden_size": 576,
  "initializer_range": 0.041666666666666664,
  "intermediate_size": 1536,
  "is_llama_config": true,
  "max_position_embeddings": 8192,
  "mlp_bias": false,
  "model_type": "llama",
  "num_attention_heads": 9,
  "num_hidden_layers": 30,
  "num_key_value_heads": 3,
  "pretraining_tp": 1,
  "rms_norm_eps": 1e-05,
  "rope_interleaved": false,
  "rope_scaling": null,
  "rope_theta": 100000,
  "tie_word_embeddings": true,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.45.1",
  "use_cache": true,
  "vocab_size": 49152
}

LlamaForCausalLM(
  (model): LlamaModel(
    (embed_tokens): Embedding(49152, 576)
    (layers): ModuleList(
      (0-29): 30 x LlamaDecoderLayer(
        (self_attn): LlamaSdpaAttention(
          (q_proj): Linear(in_features=576, out_features=576, bias=False)
          (k_proj): Linear(in_features=576, out_features=192, bias=False)
          (v_proj): Linear(in_features=576, out_features=192, bias=False)
          (o_proj): Linear(in_features=576, out_features=576, bias=False)
          (rotary_emb): LlamaRotaryEmbedding()
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=576, out_features=1536, bias=False)
          (up_proj): Linear(in_features=576, out_features=1536, bias=False)
          (down_proj): Linear(in_features=1536, out_features=576, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): LlamaRMSNorm((576,), eps=1e-05)
        (post_attention_layernorm): LlamaRMSNorm((576,), eps=1e-05)
      )
    )
    (norm): LlamaRMSNorm((576,), eps=1e-05)
    (rotary_emb): LlamaRotaryEmbedding()
  )
  (lm_head): Linear(in_features=576, out_features=49152, bias=False)
)
GPT2TokenizerFast(name_or_path='HuggingFaceTB/SmolLM2-135M', vocab_size=49152, model_max_length=8192, is_fast=True, padding_side='right', truncation_side='right', special_tokens={'bos_token': '<|endoftext|>', 'eos_token': '<|endoftext|>', 'unk_token': '<|endoftext|>', 'additional_special_tokens': ['<|endoftext|>', '<|im_start|>', '<|im_end|>', '<repo_name>', '<reponame>', '<file_sep>', '<filename>', '<gh_stars>', '<issue_start>', '<issue_comment>', '<issue_closed>', '<jupyter_start>', '<jupyter_text>', '<jupyter_code>', '<jupyter_output>', '<jupyter_script>', '<empty_output>']}, clean_up_tokenization_spaces=False),  added_tokens_decoder={
	0: AddedToken("<|endoftext|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	1: AddedToken("<|im_start|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	2: AddedToken("<|im_end|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	3: AddedToken("<repo_name>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	4: AddedToken("<reponame>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	5: AddedToken("<file_sep>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	6: AddedToken("<filename>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	7: AddedToken("<gh_stars>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	8: AddedToken("<issue_start>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	9: AddedToken("<issue_comment>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	10: AddedToken("<issue_closed>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	11: AddedToken("<jupyter_start>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	12: AddedToken("<jupyter_text>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	13: AddedToken("<jupyter_code>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	14: AddedToken("<jupyter_output>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	15: AddedToken("<jupyter_script>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	16: AddedToken("<empty_output>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
}
The attention mask and the pad token id were not set. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
Setting `pad_token_id` to `eos_token_id`:None for open-end generation.
The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
```