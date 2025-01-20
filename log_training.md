```bash
% python3 train.py              
Note: Environment variable`HF_TOKEN` is set and is the current active token independently from the token you've just configured.
Successfully logged in to Hugging Face.
SmolLM2ForCausalLM(
  (model): SmolLM2Model(
    (embed_tokens): Embedding(49152, 576)
    (layers): ModuleList(
      (0-29): 30 x SmolLM2Block(
        (input_layernorm): RMSNorm()
        (attention): SmolLM2Attention(
          (q_proj): Linear(in_features=576, out_features=576, bias=False)
          (k_proj): Linear(in_features=576, out_features=192, bias=False)
          (v_proj): Linear(in_features=576, out_features=192, bias=False)
          (o_proj): Linear(in_features=576, out_features=576, bias=False)
        )
        (post_attention_layernorm): RMSNorm()
        (mlp): SmolLM2MLP(
          (gate_proj): Linear(in_features=576, out_features=1536, bias=False)
          (up_proj): Linear(in_features=576, out_features=1536, bias=False)
          (down_proj): Linear(in_features=1536, out_features=576, bias=False)
          (act_fn): SiLU()
        )
      )
    )
    (norm): RMSNorm()
  )
  (lm_head): Linear(in_features=576, out_features=49152, bias=False)
)
Model parameters: 134515008
Using device: mps
Resolving data files: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 104/104 [00:01<00:00, 94.71it/s]
Resolving data files: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 104/104 [00:00<00:00, 389471.09it/s]

Verifying data streaming...
Data loading successful
Training:   0%|                                                                                                                                                     | 0/5000 [00:00<?, ?it/s]Starting training...
/path/to/model.py:209: UserWarning: Casting complex values to real discards the imaginary part (Triggered internally at /Users/runner/work/pytorch/pytorch/pytorch/aten/src/ATen/native/Copy.cpp:305.)
  freqs_cis = self.freqs_cis.to(device=hidden_states.device, dtype=hidden_states.dtype)

Step 0
Loss: 11.3210
LR: 0.000300
Unique content: 0
Metrics saved to checkpoints/metrics_0.pt
Training:   2%|██▋                                                                                                                                      | 100/5000 [05:36<4:19:53,  3.18s/it]
Step 100
Loss: 6.0486
LR: 0.000300
Unique content: 3199
Metrics saved to checkpoints/metrics_100.pt
Training:   4%|█████▍                                                                                                                                   | 200/5000 [10:47<4:02:54,  3.04s/it]
Step 200
Loss: 2.8005
LR: 0.000300
Unique content: 6399
Metrics saved to checkpoints/metrics_200.pt
Training:   6%|████████▏                                                                                                                                | 300/5000 [15:48<3:49:44,  2.93s/it]
Step 300
Loss: 1.0425
LR: 0.000300
Unique content: 9597
Metrics saved to checkpoints/metrics_300.pt
Training:   8%|██████████▉                                                                                                                              | 400/5000 [20:48<3:50:44,  3.01s/it]
Step 400
Loss: 0.3683
LR: 0.000300
Unique content: 12795
Metrics saved to checkpoints/metrics_400.pt
Training:  10%|█████████████▋                                                                                                                           | 500/5000 [25:49<3:45:38,  3.01s/it]
Step 500
Loss: 0.1768
LR: 0.000300
Unique content: 15993
Metrics saved to checkpoints/metrics_500.pt

Generating with temperature: 0.8

Generated (500 steps):
<< Once upon a time  events Richard G financial St KPPPP The_ today_____________ =_
   __`[ =_______________
            >>

Generating with temperature: 0.8

Generated (500 steps):
<< The scientific method  function early two while early regardless3 evenH a Americans a materials a own can asked had asked otherwise carefully carefully carefully carefully carefully carefully carefully carefully carefully carefully carefully carefully carefully carefully carefully carefully carefully carefully carefully yourself carefully yourself yourself other colorful colonial philosopher roll roll salaries >>
Checkpoint saved at step 500
Model and tokenizer saved to smollm2_model_step_500
Training:  12%|████████████████▍                                                                                                                        | 600/5000 [30:55<3:34:36,  2.93s/it]
Step 600
Loss: 0.1194
LR: 0.000300
Unique content: 19190
Metrics saved to checkpoints/metrics_600.pt
Training:  14%|███████████████████▏                                                                                                                     | 700/5000 [35:49<3:29:58,  2.93s/it]
Step 700
Loss: 0.0681
LR: 0.000300
Unique content: 22380
Metrics saved to checkpoints/metrics_700.pt
Training:  16%|█████████████████████▉                                                                                                                   | 800/5000 [40:43<3:26:28,  2.95s/it]
Step 800
Loss: 0.0652
LR: 0.000300
Unique content: 25576
Metrics saved to checkpoints/metrics_800.pt
Training:  18%|████████████████████████▋                                                                                                                | 900/5000 [45:37<3:20:46,  2.94s/it]
Step 900
Loss: 0.0424
LR: 0.000300
Unique content: 28772
Metrics saved to checkpoints/metrics_900.pt
Training:  20%|███████████████████████████▏                                                                                                            | 1000/5000 [50:31<3:14:57,  2.92s/it]
Step 1000
Loss: 0.0270
LR: 0.000300
Unique content: 31968
Metrics saved to checkpoints/metrics_1000.pt

Generating with temperature: 0.8

Generated (1000 steps):
<< Once upon a time 
       y
       FQStep\ / ( ( (                                        >>

Generating with temperature: 0.8

Generated (1000 steps):
<< The scientific method 0 friends resources people has anything has anything has any merely merely merely merely merely merely merely merely merely merely merely merely merely well well interest corporationsol resources AreC OperationsC555 ( ( ( (           >>
Checkpoint saved at step 1000
Model and tokenizer saved to smollm2_model_step_1000
Training:  22%|█████████████████████████████▉                                                                                                          | 1100/5000 [55:35<3:09:42,  2.92s/it]
Step 1100
Loss: 0.0183
LR: 0.000300
Unique content: 35156
Metrics saved to checkpoints/metrics_1100.pt
Training:  24%|████████████████████████████████▏                                                                                                     | 1200/5000 [1:00:28<3:03:58,  2.90s/it]
Step 1200
Loss: 0.0378
LR: 0.000300
Unique content: 38350
Metrics saved to checkpoints/metrics_1200.pt
Training:  26%|██████████████████████████████████▊                                                                                                   | 1300/5000 [1:05:20<2:59:10,  2.91s/it]
Step 1300
Loss: 0.0077
LR: 0.000300
Unique content: 41545
Metrics saved to checkpoints/metrics_1300.pt
Training:  28%|█████████████████████████████████████▌                                                                                                | 1400/5000 [1:10:12<2:54:15,  2.90s/it]
Step 1400
Loss: 0.0067
LR: 0.000300
Unique content: 44732
Metrics saved to checkpoints/metrics_1400.pt
Training:  30%|████████████████████████████████████████▏                                                                                             | 1500/5000 [1:15:04<2:49:53,  2.91s/it]
Step 1500
Loss: 0.0092
LR: 0.000300
Unique content: 47922
Metrics saved to checkpoints/metrics_1500.pt

Generating with temperature: 0.8

Generated (1500 steps):
<< Once upon a time  subto Dto Din D doing ratherling:ling::::: restir lens Health Health Health Health Health G Health G G G G G G G G G G G G G G G G G G G G G G G >>

Generating with temperature: 0.8

Generated (1500 steps):
<< The scientific method  it number it3E3 person3333
3
                                     >>
Checkpoint saved at step 1500
Model and tokenizer saved to smollm2_model_step_1500
Training:  32%|██████████████████████████████████████████▉                                                                                           | 1600/5000 [1:20:02<2:43:51,  2.89s/it]
Step 1600
Loss: 0.0063
LR: 0.000300
Unique content: 51113
Metrics saved to checkpoints/metrics_1600.pt
Training:  34%|█████████████████████████████████████████████▌                                                                                        | 1700/5000 [1:24:54<2:39:09,  2.89s/it]
Step 1700
Loss: 0.0078
LR: 0.000300
Unique content: 54301
Metrics saved to checkpoints/metrics_1700.pt
Training:  36%|████████████████████████████████████████████████▏                                                                                     | 1800/5000 [1:29:46<2:33:35,  2.88s/it]
Step 1800
Loss: 0.0072
LR: 0.000300
Unique content: 57478
Metrics saved to checkpoints/metrics_1800.pt
Training:  38%|██████████████████████████████████████████████████▉                                                                                   | 1900/5000 [1:34:37<2:29:53,  2.90s/it]
Step 1900
Loss: 0.0023
LR: 0.000300
Unique content: 60669
Metrics saved to checkpoints/metrics_1900.pt
Training:  40%|█████████████████████████████████████████████████████▌                                                                                | 2000/5000 [1:39:27<2:24:47,  2.90s/it]
Step 2000
Loss: 0.0038
LR: 0.000300
Unique content: 63859
Metrics saved to checkpoints/metrics_2000.pt

Generating with temperature: 0.8

Generated (2000 steps):
<< Once upon a time  surroundingright).right.............................................. >>

Generating with temperature: 0.8

Generated (2000 steps):
<< The scientific method 1 through" emergeduan commerce disciplineoodle disciplinebeam Effortsickness Behavior"), Behavior Opin dramat ceramicInsteadh Together












V111111111111111 >>
Checkpoint saved at step 2000
Model and tokenizer saved to smollm2_model_step_2000
Training:  42%|████████████████████████████████████████████████████████▎                                                                             | 2100/5000 [1:44:26<2:19:31,  2.89s/it]
Step 2100
Loss: 0.0029
LR: 0.000300
Unique content: 67047
Metrics saved to checkpoints/metrics_2100.pt
Training:  44%|██████████████████████████████████████████████████████████▉                                                                           | 2200/5000 [1:49:17<2:14:41,  2.89s/it]
Step 2200
Loss: 0.0075
LR: 0.000300
Unique content: 70240
Metrics saved to checkpoints/metrics_2200.pt
Training:  46%|█████████████████████████████████████████████████████████████▋                                                                        | 2300/5000 [1:54:09<2:10:24,  2.90s/it]
Step 2300
Loss: 0.0024
LR: 0.000300
Unique content: 73436
Metrics saved to checkpoints/metrics_2300.pt
Training:  48%|████████████████████████████████████████████████████████████████▎                                                                     | 2400/5000 [1:58:59<2:05:11,  2.89s/it]
Step 2400
Loss: 0.0095
LR: 0.000300
Unique content: 76628
Metrics saved to checkpoints/metrics_2400.pt
Training:  50%|███████████████████████████████████████████████████████████████████                                                                   | 2500/5000 [2:03:49<2:00:52,  2.90s/it]
Step 2500
Loss: 0.0012
LR: 0.000300
Unique content: 79814
Metrics saved to checkpoints/metrics_2500.pt

Generating with temperature: 0.8

Generated (2500 steps):
<< Once upon a time  Intertode-de-de-devvvvvvvvv signsto =_____________________________ >>

Generating with temperature: 0.8

Generated (2500 steps):
<< The scientific method -1 statistical
=
=
=
=
=




































 >>
Checkpoint saved at step 2500
Model and tokenizer saved to smollm2_model_step_2500
Training:  52%|█████████████████████████████████████████████████████████████████████▋                                                                | 2600/5000 [2:08:48<1:56:31,  2.91s/it]
Step 2600
Loss: 0.0081
LR: 0.000300
Unique content: 83006
Metrics saved to checkpoints/metrics_2600.pt
Training:  54%|████████████████████████████████████████████████████████████████████████▎                                                             | 2700/5000 [2:13:38<1:50:47,  2.89s/it]
Step 2700
Loss: 0.0067
LR: 0.000300
Unique content: 86198
Metrics saved to checkpoints/metrics_2700.pt
Training:  56%|███████████████████████████████████████████████████████████████████████████                                                           | 2800/5000 [2:18:29<1:46:13,  2.90s/it]
Step 2800
Loss: 0.0051
LR: 0.000300
Unique content: 89385
Metrics saved to checkpoints/metrics_2800.pt
Training:  58%|█████████████████████████████████████████████████████████████████████████████▋                                                        | 2900/5000 [2:23:20<1:41:07,  2.89s/it]
Step 2900
Loss: 0.0020
LR: 0.000300
Unique content: 92565
Metrics saved to checkpoints/metrics_2900.pt
Training:  60%|████████████████████████████████████████████████████████████████████████████████▍                                                     | 3000/5000 [2:28:11<1:37:06,  2.91s/it]
Step 3000
Loss: 0.0054
LR: 0.000300
Unique content: 95754
Metrics saved to checkpoints/metrics_3000.pt

Generating with temperature: 0.8

Generated (3000 steps):
<< Once upon a time [_{[_{[_{ \ \():):):):):):}}}}}}}}}}}}}}}}}}}(_______________ >>

Generating with temperature: 0.8

Generated (3000 steps):
<< The scientific method _9P1111111111to1C111Kreal Itles Understanding IntersectionalitymedinXaxsssssssssssasasasasasasasasasas >>
Checkpoint saved at step 3000
Model and tokenizer saved to smollm2_model_step_3000
Training:  62%|███████████████████████████████████████████████████████████████████████████████████                                                   | 3100/5000 [2:33:09<1:31:42,  2.90s/it]
Step 3100
Loss: 0.0026
LR: 0.000300
Unique content: 98938
Metrics saved to checkpoints/metrics_3100.pt
Training:  64%|█████████████████████████████████████████████████████████████████████████████████████▊                                                | 3200/5000 [2:38:00<1:26:34,  2.89s/it]
Step 3200
Loss: 0.0007
LR: 0.000300
Unique content: 102123
Metrics saved to checkpoints/metrics_3200.pt
Training:  66%|████████████████████████████████████████████████████████████████████████████████████████▍                                             | 3300/5000 [2:42:50<1:22:00,  2.89s/it]
Step 3300
Loss: 0.0041
LR: 0.000300
Unique content: 105307
Metrics saved to checkpoints/metrics_3300.pt
Training:  68%|███████████████████████████████████████████████████████████████████████████████████████████                                           | 3400/5000 [2:47:41<1:18:00,  2.93s/it]
Step 3400
Loss: 0.0011
LR: 0.000300
Unique content: 108500
Metrics saved to checkpoints/metrics_3400.pt
Training:  70%|█████████████████████████████████████████████████████████████████████████████████████████████▊                                        | 3500/5000 [2:52:32<1:12:02,  2.88s/it]
Step 3500
Loss: 0.0009
LR: 0.000300
Unique content: 111688
Metrics saved to checkpoints/metrics_3500.pt

Generating with temperature: 0.8

Generated (3500 steps):
<< Once upon a time  upon andCourse and we and we and and and and and and and and and and and and and and and and and and and and and and and and and n low customs Passion Party Bahamas Party team Party Party Party PartyRAYhelp toys flexibility flexibility flexibility >>

Generating with temperature: 0.8

Generated (3500 steps):
<< The scientific method 1 Notably/H/to \class hardupepeEEE E E $ E $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $  **1
1
1
1
1
1
 >>
Checkpoint saved at step 3500
Model and tokenizer saved to smollm2_model_step_3500
Training:  72%|████████████████████████████████████████████████████████████████████████████████████████████████▍                                     | 3600/5000 [2:57:30<1:07:19,  2.89s/it]
Step 3600
Loss: 0.0015
LR: 0.000300
Unique content: 114870
Metrics saved to checkpoints/metrics_3600.pt
Training:  74%|███████████████████████████████████████████████████████████████████████████████████████████████████▏                                  | 3700/5000 [3:02:20<1:03:08,  2.91s/it]
Step 3700
Loss: 0.0006
LR: 0.000300
Unique content: 118055
Metrics saved to checkpoints/metrics_3700.pt
Training:  76%|███████████████████████████████████████████████████████████████████████████████████████████████████████▎                                | 3800/5000 [3:07:11<57:42,  2.89s/it]
Step 3800
Loss: 0.0010
LR: 0.000300
Unique content: 121231
Metrics saved to checkpoints/metrics_3800.pt
Training:  78%|██████████████████████████████████████████████████████████████████████████████████████████████████████████                              | 3900/5000 [3:12:02<53:21,  2.91s/it]
Step 3900
Loss: 0.0008
LR: 0.000300
Unique content: 124407
Metrics saved to checkpoints/metrics_3900.pt
Training:  80%|████████████████████████████████████████████████████████████████████████████████████████████████████████████▊                           | 4000/5000 [3:16:53<48:02,  2.88s/it]
Step 4000
Loss: 0.0015
LR: 0.000300
Unique content: 127587
Metrics saved to checkpoints/metrics_4000.pt

Generating with temperature: 0.8

Generated (4000 steps):
<< Once upon a time  theirZ TwZyy per per ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( \ \ \ \ \ \ \ \ \ \ \ \ \ \[[$$$$ >>

Generating with temperature: 0.8

Generated (4000 steps):
<< The scientific method 1Coursefire More More richer coordinated enterprisesitu façadeIS):** Cur musculoskeletalininininininininininininininininininininininininininininininininininw8 >>
Checkpoint saved at step 4000
Model and tokenizer saved to smollm2_model_step_4000
Training:  82%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████▌                        | 4100/5000 [3:21:51<43:29,  2.90s/it]
Step 4100
Loss: 0.0015
LR: 0.000300
Unique content: 130773
Metrics saved to checkpoints/metrics_4100.pt
Training:  84%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏                     | 4200/5000 [3:26:41<38:32,  2.89s/it]
Step 4200
Loss: 0.0006
LR: 0.000300
Unique content: 133952
Metrics saved to checkpoints/metrics_4200.pt
Training:  86%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉                   | 4300/5000 [3:31:32<33:49,  2.90s/it]
Step 4300
Loss: 0.0002
LR: 0.000300
Unique content: 137130
Metrics saved to checkpoints/metrics_4300.pt
Training:  88%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋                | 4400/5000 [3:36:23<28:49,  2.88s/it]
Step 4400
Loss: 0.0016
LR: 0.000300
Unique content: 140309
Metrics saved to checkpoints/metrics_4400.pt
Training:  90%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍             | 4500/5000 [3:41:14<23:59,  2.88s/it]
Step 4500
Loss: 0.0016
LR: 0.000300
Unique content: 143489
Metrics saved to checkpoints/metrics_4500.pt

Generating with temperature: 0.8

Generated (4500 steps):
<< Once upon a time 
3_{3_{_{_{_{_{_{_{_{_{_{}}}}}}}}}}}}}}}}}_{_{_{X 











-- >>

Generating with temperature: 0.8

Generated (4500 steps):
<< The scientific method -asion
 legislatures
 Dogs____________________________________________ >>
Checkpoint saved at step 4500
Model and tokenizer saved to smollm2_model_step_4500
Training:  92%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████           | 4600/5000 [3:46:12<19:23,  2.91s/it]
Step 4600
Loss: 0.0007
LR: 0.000300
Unique content: 146672
Metrics saved to checkpoints/metrics_4600.pt
Training:  94%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊        | 4700/5000 [3:51:02<14:25,  2.88s/it]
Step 4700
Loss: 0.0031
LR: 0.000300
Unique content: 149853
Metrics saved to checkpoints/metrics_4700.pt
Training:  96%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌     | 4800/5000 [3:55:52<09:36,  2.88s/it]
Step 4800
Loss: 0.0045
LR: 0.000300
Unique content: 153035
Metrics saved to checkpoints/metrics_4800.pt
Training:  98%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎  | 4900/5000 [4:00:42<04:52,  2.92s/it]
Step 4900
Loss: 0.0009
LR: 0.000300
Unique content: 156214
Metrics saved to checkpoints/metrics_4900.pt
Training: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5000/5000 [4:05:34<00:00,  2.88s/it]Metrics saved to checkpoints/metrics_final.pt
Checkpoint saved at step 5000
Model and tokenizer saved to smollm2_model_final

Starting additional training...

Loading final checkpoint and training for 50 more steps...
/path/to/train.py:124: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  checkpoint = torch.load(path)
Additional step 0, Loss: 0.0006
Additional step 1, Loss: 0.0005
Additional step 2, Loss: 0.0008
Additional step 3, Loss: 0.0006
Additional step 4, Loss: 0.0004
Additional step 5, Loss: 0.0007
Additional step 6, Loss: 0.0004
Additional step 7, Loss: 0.0010
Additional step 8, Loss: 0.0005
Additional step 9, Loss: 0.0011
Additional step 10, Loss: 0.0010
Additional step 11, Loss: 0.0004
Additional step 12, Loss: 0.0007
Additional step 13, Loss: 0.0006
Additional step 14, Loss: 0.0007
Additional step 15, Loss: 0.0017
Additional step 16, Loss: 0.0007
Additional step 17, Loss: 0.0005
Additional step 18, Loss: 0.0006
Additional step 19, Loss: 0.0007
Additional step 20, Loss: 0.0005
Additional step 21, Loss: 0.0005
Additional step 22, Loss: 0.0010
Additional step 23, Loss: 0.0006
Additional step 24, Loss: 0.0005
Additional step 25, Loss: 0.0004
Additional step 26, Loss: 0.0006
Additional step 27, Loss: 0.0013
Additional step 28, Loss: 0.0005
Additional step 29, Loss: 0.0014
Additional step 30, Loss: 0.0005
Additional step 31, Loss: 0.0004
Additional step 32, Loss: 0.0022
Additional step 33, Loss: 0.0011
Additional step 34, Loss: 0.0006
Additional step 35, Loss: 0.0004
Additional step 36, Loss: 0.0012
Additional step 37, Loss: 0.0007
Additional step 38, Loss: 0.0004
Additional step 39, Loss: 0.0015
Additional step 40, Loss: 0.0004
Additional step 41, Loss: 0.0006
Additional step 42, Loss: 0.0006
Additional step 43, Loss: 0.0006
Additional step 44, Loss: 0.0007
Additional step 45, Loss: 0.0006
Additional step 46, Loss: 0.0006
Additional step 47, Loss: 0.0007
Additional step 48, Loss: 0.0005
Additional step 49, Loss: 0.0010
Metrics saved to checkpoints/metrics_additional.pt
Checkpoint saved at step 5050
Training completed successfully!
Training: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5000/5000 [4:08:04<00:00,  2.98s/it]
```