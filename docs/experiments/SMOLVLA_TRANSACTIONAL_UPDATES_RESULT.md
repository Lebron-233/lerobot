# F-TUA1：全训练集有限步接受与完整回退

execution HEAD：`dc07ccb58a88fd1c7f29708d9fa248d8a3554200`；运行`completed`；独立接纳`True`。

原72训练/18episode用于每次接受检查；16开发验证/4episode仅初始exact重放和最终评估，非独立资格。
72次同序AdamW提案，拒绝同时回退参数、moment和step；不是调小学习率或追加搜索。
历史BRP/guarded/plain是此前已审计对照，不是本轮同期重训。oracle是未来视觉冻结策略，不是专家。

接受3，拒绝69；开发推进`False`。
训练检查通过由接受规则直接约束，本身不能用作泛化或安全成功证据。

## 训练接受检查

```json
{
  "initial": {
    "objective": 2.4790027436206663,
    "row0": 0.053600007189489124,
    "chunk": 0.08274325003580583,
    "max_excess": 0.6427515004625787
  },
  "final": {
    "objective": 2.4467568542627367,
    "row0": 0.053117466641086936,
    "chunk": 0.08081370471900945,
    "max_excess": 0.6339000328909978
  }
}
```

## train（episode等权）

| 条件 | N/episode | 首动作MSE | chunk MSE | token MSE |
|---|---:|---:|---:|---:|
| brp_mismatched | 69/18 | 0.113423646107 | 0.092605483268 | 2323.494624177 |
| brp_true | 72/18 | 0.060416054279 | 0.079370001134 | 2313.659300061 |
| brp_zero | 72/18 | 0.075968549259 | 0.088663766316 | 2312.744809170 |
| frozen | 72/18 | 0.053600007189 | 0.082743250036 | 2314.359247736 |
| guarded_mismatched | 69/18 | 0.113131045002 | 0.089997832693 | 2323.227724392 |
| guarded_true | 72/18 | 0.062139924135 | 0.079646371177 | 2313.473796063 |
| guarded_zero | 72/18 | 0.075968549259 | 0.088663766316 | 2312.744809170 |
| identity | 72/18 | 0.075968549259 | 0.088663766316 | 2312.744809170 |
| old_centered | 72/18 | 0.055629032020 | 0.079294967078 | 2315.185317077 |
| plain_mismatched | 69/18 | 0.109606586626 | 0.088156931701 | 2323.036680815 |
| plain_true | 72/18 | 0.063058823156 | 0.080341232542 | 2313.366521975 |
| plain_zero | 72/18 | 0.075968549259 | 0.088663766316 | 2312.744809170 |
| transaction_mismatched | 69/18 | 0.116481795575 | 0.092608131555 | 2324.262661072 |
| transaction_true | 72/18 | 0.053117466641 | 0.080813704719 | 2314.243698717 |
| transaction_zero | 72/18 | 0.075968549259 | 0.088663766316 | 2312.744809170 |

下列对比中的候选为transaction；历史BRP指标单独标作brp。

相对identity：N=72，均值收益0.022851082618，样本{'worsened': 21, 'improved': 51}，episode改善15。
相对frozen：N=72，均值收益0.000482540548，样本{'improved': 46, 'worsened': 26}，episode改善12。
相对plain_true：N=72，均值收益0.009941356515，样本{'improved': 35, 'worsened': 37}，episode改善11。
相对guarded_true：N=72，均值收益0.009022457494，样本{'improved': 35, 'worsened': 37}，episode改善9。
相对transaction_mismatched：N=69，均值收益0.059739747940，样本{'improved': 51, 'worsened': 18}，episode改善17。

相对冻结候选最不利样本（不删例）：
[1, 49, 5]: -0.006507316917 (worsened)
[5, 46, 6]: -0.004441039041 (worsened)
[1, 48, 3]: -0.003197005433 (worsened)
[1, 48, 5]: -0.002731974067 (worsened)
[3, 48, 5]: -0.002328050399 (worsened)

留一episode收益：`{"0/46": 0.0005292414021209207, "0/48": 0.0005012683044536072, "0/49": 0.0005078175434939517, "1/46": 0.00046206354354992515, "1/48": 0.0005860935357014077, "1/49": 0.000544110731588991, "2/46": 0.0003326643499259106, "2/48": 0.0004929623471492446, "2/49": 0.0003642517787813663, "3/46": 0.0004714294108713014, "3/48": 0.0004719900381184397, "3/49": 0.0005053696649449398, "4/46": 0.00041056846977477143, "4/48": 0.0004123164547350756, "4/49": 0.0005208313023288958, "5/46": 0.0005565279976064056, "5/48": 0.0005266063941193666, "5/49": 0.000489616601974591}`。

## validation（episode等权）

| 条件 | N/episode | 首动作MSE | chunk MSE | token MSE |
|---|---:|---:|---:|---:|
| brp_mismatched | 16/4 | 0.040417199676 | 0.063237685805 | 2673.065829055 |
| brp_true | 16/4 | 0.014130978751 | 0.028661323583 | 2671.731253134 |
| brp_zero | 16/4 | 0.031709702433 | 0.053248021229 | 2670.828930525 |
| frozen | 16/4 | 0.012048372598 | 0.026728824581 | 2672.575387909 |
| guarded_mismatched | 16/4 | 0.039137933789 | 0.062022989630 | 2672.748383999 |
| guarded_true | 16/4 | 0.015590678394 | 0.028277092513 | 2671.532252757 |
| guarded_zero | 16/4 | 0.031709702433 | 0.053248021229 | 2670.828930525 |
| identity | 16/4 | 0.031709702433 | 0.053248021229 | 2670.828930525 |
| old_centered | 16/4 | 0.013862918700 | 0.021462120248 | 2673.353070642 |
| plain_mismatched | 16/4 | 0.037120900978 | 0.060825469549 | 2672.528362681 |
| plain_true | 16/4 | 0.015780598674 | 0.031444426151 | 2671.456312184 |
| plain_zero | 16/4 | 0.031709702433 | 0.053248021229 | 2670.828930525 |
| transaction_mismatched | 16/4 | 0.043593465424 | 0.065290471691 | 2673.962999958 |
| transaction_true | 16/4 | 0.012577338758 | 0.027514487743 | 2672.451256429 |
| transaction_zero | 16/4 | 0.031709702433 | 0.053248021229 | 2670.828930525 |

下列对比中的候选为transaction；历史BRP指标单独标作brp。

相对identity：N=16，均值收益0.019132363675，样本{'improved': 12, 'worsened': 4}，episode改善4。
相对frozen：N=16，均值收益-0.000528966160，样本{'worsened': 11, 'improved': 5}，episode改善0。
相对plain_true：N=16，均值收益0.003203259916，样本{'worsened': 8, 'improved': 8}，episode改善2。
相对guarded_true：N=16，均值收益0.003013339635，样本{'worsened': 8, 'improved': 8}，episode改善3。
相对transaction_mismatched：N=16，均值收益0.031016126666，样本{'worsened': 10, 'improved': 6}，episode改善4。

相对冻结候选最不利样本（不删例）：
[6, 48, 5]: -0.003547926154 (worsened)
[6, 49, 3]: -0.002116804648 (worsened)
[7, 49, 5]: -0.002116266838 (worsened)
[7, 48, 6]: -0.000597846584 (worsened)
[7, 48, 3]: -0.000589815426 (worsened)

留一episode收益：`{"6/48": -0.0005350371098052613, "6/49": -0.0004907661489631371, "7/48": -0.000623150367981969, "7/49": -0.0004669110125154052}`。

## 更新接受日志

| 提案 | 样本 | 接受 | 提议row0均值 | 保留row0均值 | 未满足检查 |
|---:|---|---|---:|---:|---|
| 1 | [0, 46, 3] | True | 0.053432636 | 0.053432636 |  |
| 2 | [1, 46, 3] | False | 0.053070939 | 0.053432636 | objective_decreases, chunk_not_worse_incumbent |
| 3 | [2, 46, 3] | False | 0.053485455 | 0.053432636 | row0_not_worse_incumbent |
| 4 | [3, 46, 3] | True | 0.053160431 | 0.053160431 |  |
| 5 | [4, 46, 3] | False | 0.053519305 | 0.053160431 | objective_decreases, row0_not_worse_incumbent, chunk_not_worse_incumbent |
| 6 | [5, 46, 3] | False | 0.053627134 | 0.053160431 | objective_decreases, row0_not_worse_incumbent, row0_not_worse_initial, chunk_not_worse_incumbent, max_excess_not_worse_incumbent |
| 7 | [0, 48, 3] | False | 0.053371849 | 0.053160431 | objective_decreases, row0_not_worse_incumbent, chunk_not_worse_incumbent |
| 8 | [1, 48, 3] | False | 0.053413299 | 0.053160431 | objective_decreases, row0_not_worse_incumbent, chunk_not_worse_incumbent, max_excess_not_worse_incumbent |
| 9 | [2, 48, 3] | False | 0.053319574 | 0.053160431 | objective_decreases, row0_not_worse_incumbent, chunk_not_worse_incumbent, max_excess_not_worse_incumbent |
| 10 | [3, 48, 3] | False | 0.053298054 | 0.053160431 | objective_decreases, row0_not_worse_incumbent, chunk_not_worse_incumbent |
| 11 | [4, 48, 3] | False | 0.053141323 | 0.053160431 | objective_decreases, chunk_not_worse_incumbent |
| 12 | [5, 48, 3] | False | 0.053470847 | 0.053160431 | row0_not_worse_incumbent, max_excess_not_worse_incumbent |
| 13 | [0, 49, 3] | False | 0.053532149 | 0.053160431 | row0_not_worse_incumbent, max_excess_not_worse_incumbent |
| 14 | [1, 49, 3] | False | 0.053458207 | 0.053160431 | row0_not_worse_incumbent |
| 15 | [2, 49, 3] | False | 0.053238175 | 0.053160431 | row0_not_worse_incumbent |
| 16 | [3, 49, 3] | False | 0.053021657 | 0.053160431 | max_excess_not_worse_incumbent |
| 17 | [4, 49, 3] | False | 0.053092035 | 0.053160431 | objective_decreases, chunk_not_worse_incumbent |
| 18 | [5, 49, 3] | False | 0.053209216 | 0.053160431 | objective_decreases, row0_not_worse_incumbent, chunk_not_worse_incumbent, max_excess_not_worse_incumbent |
| 19 | [0, 46, 4] | False | 0.053193686 | 0.053160431 | objective_decreases, row0_not_worse_incumbent, chunk_not_worse_incumbent |
| 20 | [1, 46, 4] | False | 0.053421343 | 0.053160431 | row0_not_worse_incumbent, max_excess_not_worse_incumbent |
| 21 | [2, 46, 4] | False | 0.053385619 | 0.053160431 | objective_decreases, row0_not_worse_incumbent, chunk_not_worse_incumbent, max_excess_not_worse_incumbent |
| 22 | [3, 46, 4] | False | 0.053720009 | 0.053160431 | objective_decreases, row0_not_worse_incumbent, row0_not_worse_initial, chunk_not_worse_incumbent, chunk_not_worse_initial |
| 23 | [4, 46, 4] | False | 0.053422334 | 0.053160431 | row0_not_worse_incumbent, max_excess_not_worse_incumbent |
| 24 | [5, 46, 4] | False | 0.053527646 | 0.053160431 | objective_decreases, row0_not_worse_incumbent, chunk_not_worse_incumbent |
| 25 | [0, 48, 4] | False | 0.053230778 | 0.053160431 | objective_decreases, row0_not_worse_incumbent, chunk_not_worse_incumbent, max_excess_not_worse_incumbent |
| 26 | [1, 48, 4] | False | 0.053245342 | 0.053160431 | row0_not_worse_incumbent |
| 27 | [2, 48, 4] | False | 0.053530432 | 0.053160431 | objective_decreases, row0_not_worse_incumbent, chunk_not_worse_incumbent |
| 28 | [3, 48, 4] | False | 0.053333112 | 0.053160431 | objective_decreases, row0_not_worse_incumbent, chunk_not_worse_incumbent |
| 29 | [4, 48, 4] | True | 0.053117467 | 0.053117467 |  |
| 30 | [5, 48, 4] | False | 0.053060937 | 0.053117467 | objective_decreases, chunk_not_worse_incumbent, max_excess_not_worse_incumbent |
| 31 | [0, 49, 4] | False | 0.052861830 | 0.053117467 | objective_decreases, chunk_not_worse_incumbent |
| 32 | [1, 49, 4] | False | 0.052976704 | 0.053117467 | objective_decreases, chunk_not_worse_incumbent, max_excess_not_worse_incumbent |
| 33 | [2, 49, 4] | False | 0.053009342 | 0.053117467 | objective_decreases, chunk_not_worse_incumbent, max_excess_not_worse_incumbent |
| 34 | [3, 49, 4] | False | 0.053107106 | 0.053117467 | objective_decreases, chunk_not_worse_incumbent, max_excess_not_worse_incumbent |
| 35 | [4, 49, 4] | False | 0.053575351 | 0.053117467 | objective_decreases, row0_not_worse_incumbent, max_excess_not_worse_incumbent |
| 36 | [5, 49, 4] | False | 0.052978028 | 0.053117467 | objective_decreases, chunk_not_worse_incumbent |
| 37 | [0, 46, 5] | False | 0.053425883 | 0.053117467 | objective_decreases, row0_not_worse_incumbent, chunk_not_worse_incumbent, max_excess_not_worse_incumbent |
| 38 | [1, 46, 5] | False | 0.053584992 | 0.053117467 | objective_decreases, row0_not_worse_incumbent, chunk_not_worse_incumbent, chunk_not_worse_initial |
| 39 | [2, 46, 5] | False | 0.053194720 | 0.053117467 | objective_decreases, row0_not_worse_incumbent, chunk_not_worse_incumbent, max_excess_not_worse_incumbent |
| 40 | [3, 46, 5] | False | 0.053562723 | 0.053117467 | objective_decreases, row0_not_worse_incumbent, chunk_not_worse_incumbent, max_excess_not_worse_incumbent |
| 41 | [4, 46, 5] | False | 0.053457943 | 0.053117467 | objective_decreases, row0_not_worse_incumbent, chunk_not_worse_incumbent, max_excess_not_worse_incumbent |
| 42 | [5, 46, 5] | False | 0.053283338 | 0.053117467 | objective_decreases, row0_not_worse_incumbent, chunk_not_worse_incumbent |
| 43 | [0, 48, 5] | False | 0.053303045 | 0.053117467 | objective_decreases, row0_not_worse_incumbent, max_excess_not_worse_incumbent |
| 44 | [1, 48, 5] | False | 0.053250413 | 0.053117467 | objective_decreases, row0_not_worse_incumbent |
| 45 | [2, 48, 5] | False | 0.053487873 | 0.053117467 | objective_decreases, row0_not_worse_incumbent, chunk_not_worse_incumbent, max_excess_not_worse_incumbent |
| 46 | [3, 48, 5] | False | 0.053143995 | 0.053117467 | objective_decreases, row0_not_worse_incumbent, chunk_not_worse_incumbent, max_excess_not_worse_incumbent |
| 47 | [4, 48, 5] | False | 0.053257488 | 0.053117467 | objective_decreases, row0_not_worse_incumbent, chunk_not_worse_incumbent, max_excess_not_worse_incumbent |
| 48 | [5, 48, 5] | False | 0.053020315 | 0.053117467 | objective_decreases, chunk_not_worse_incumbent, max_excess_not_worse_incumbent |
| 49 | [0, 49, 5] | False | 0.053279476 | 0.053117467 | row0_not_worse_incumbent, chunk_not_worse_incumbent |
| 50 | [1, 49, 5] | False | 0.053587216 | 0.053117467 | objective_decreases, row0_not_worse_incumbent, chunk_not_worse_incumbent, max_excess_not_worse_incumbent |
| 51 | [2, 49, 5] | False | 0.053539387 | 0.053117467 | objective_decreases, row0_not_worse_incumbent, chunk_not_worse_incumbent, max_excess_not_worse_incumbent |
| 52 | [3, 49, 5] | False | 0.053138509 | 0.053117467 | row0_not_worse_incumbent, max_excess_not_worse_incumbent |
| 53 | [4, 49, 5] | False | 0.053325510 | 0.053117467 | objective_decreases, row0_not_worse_incumbent, max_excess_not_worse_incumbent |
| 54 | [5, 49, 5] | False | 0.053324570 | 0.053117467 | row0_not_worse_incumbent, max_excess_not_worse_incumbent |
| 55 | [0, 46, 6] | False | 0.053294819 | 0.053117467 | objective_decreases, row0_not_worse_incumbent, chunk_not_worse_incumbent, max_excess_not_worse_incumbent |
| 56 | [1, 46, 6] | False | 0.053544238 | 0.053117467 | objective_decreases, row0_not_worse_incumbent, chunk_not_worse_incumbent, max_excess_not_worse_incumbent |
| 57 | [2, 46, 6] | False | 0.053477919 | 0.053117467 | objective_decreases, row0_not_worse_incumbent, chunk_not_worse_incumbent, max_excess_not_worse_incumbent |
| 58 | [3, 46, 6] | False | 0.053205734 | 0.053117467 | objective_decreases, row0_not_worse_incumbent, chunk_not_worse_incumbent, chunk_not_worse_initial |
| 59 | [4, 46, 6] | False | 0.053412473 | 0.053117467 | objective_decreases, row0_not_worse_incumbent, chunk_not_worse_incumbent, max_excess_not_worse_incumbent |
| 60 | [5, 46, 6] | False | 0.053314489 | 0.053117467 | objective_decreases, row0_not_worse_incumbent, chunk_not_worse_incumbent, max_excess_not_worse_incumbent |
| 61 | [0, 48, 6] | False | 0.053385392 | 0.053117467 | objective_decreases, row0_not_worse_incumbent, chunk_not_worse_incumbent |
| 62 | [1, 48, 6] | False | 0.053386207 | 0.053117467 | objective_decreases, row0_not_worse_incumbent, chunk_not_worse_incumbent, chunk_not_worse_initial, max_excess_not_worse_incumbent |
| 63 | [2, 48, 6] | False | 0.053336979 | 0.053117467 | objective_decreases, row0_not_worse_incumbent, chunk_not_worse_incumbent, max_excess_not_worse_incumbent |
| 64 | [3, 48, 6] | False | 0.053431034 | 0.053117467 | objective_decreases, row0_not_worse_incumbent, chunk_not_worse_incumbent, max_excess_not_worse_incumbent |
| 65 | [4, 48, 6] | False | 0.053117376 | 0.053117467 | objective_decreases, chunk_not_worse_incumbent, max_excess_not_worse_incumbent |
| 66 | [5, 48, 6] | False | 0.053062142 | 0.053117467 | objective_decreases, chunk_not_worse_incumbent, max_excess_not_worse_incumbent |
| 67 | [0, 49, 6] | False | 0.053323820 | 0.053117467 | objective_decreases, row0_not_worse_incumbent, chunk_not_worse_incumbent, max_excess_not_worse_incumbent |
| 68 | [1, 49, 6] | False | 0.053527262 | 0.053117467 | objective_decreases, row0_not_worse_incumbent, chunk_not_worse_incumbent |
| 69 | [2, 49, 6] | False | 0.053447310 | 0.053117467 | objective_decreases, row0_not_worse_incumbent, chunk_not_worse_incumbent, max_excess_not_worse_incumbent |
| 70 | [3, 49, 6] | False | 0.053131439 | 0.053117467 | objective_decreases, row0_not_worse_incumbent, max_excess_not_worse_incumbent |
| 71 | [4, 49, 6] | False | 0.053105929 | 0.053117467 | objective_decreases, chunk_not_worse_incumbent, max_excess_not_worse_incumbent |
| 72 | [5, 49, 6] | False | 0.053539539 | 0.053117467 | objective_decreases, row0_not_worse_incumbent, chunk_not_worse_incumbent, max_excess_not_worse_incumbent |

## 固定判据与执行

```json
{
  "checks": {
    "row0_better_identity": true,
    "row0_better_transaction_mismatched": true,
    "row0_not_worse_frozen": false,
    "chunk_better_frozen": false,
    "train_worst_better_frozen": true,
    "chunk_better_guarded_true": true,
    "train_worst_better_guarded_true": false,
    "chunk_better_plain_true": true,
    "train_worst_better_plain_true": false,
    "three_of_four_episodes": true,
    "sample_majority": true,
    "validation_worst_not_worse_frozen": true,
    "nonzero_committed_updates": true
  },
  "counts": {
    "vla_loads": 1,
    "predictor_loads": 1,
    "predictor": 11210,
    "decoder": 5781,
    "identity_exact": 88,
    "old_centered_exact": 88,
    "frozen_exact": 88,
    "gradient_decoder": 72,
    "backward": 72,
    "proposals": 72,
    "accepted": 3,
    "rejected": 69,
    "zero_exact": 88,
    "captures": 448
  },
  "phases": {
    "started": 5857,
    "returned": 5857
  },
  "execution": {
    "child_pid": 655136,
    "child_exit_code": 0,
    "exit_confirmed": true,
    "started_at_utc": "2026-09-17T01:22:35.121294+00:00",
    "finished_at_utc": "2026-09-17T01:38:32.946317+00:00",
    "wall_seconds": 957.8250378409866,
    "stop_reason": null,
    "forced_termination": false,
    "pending": {},
    "active": {}
  }
}
```

全训练集扫描增加离线训练开销；不能用于线上需要未知future/oracle的风险门。
没有新Env、图像编码、资格集或真机。无部署/时延/成功率结论；旧R1阴性保持。
CPU审计不重新求梯度或运行模型；验证保存指标、门判读及优化器更新与完整回退算术。
