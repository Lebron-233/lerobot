# F-IAR1：固定残差的 identity 基底置换

execution HEAD: `1f74c04d2ce9d02562d12e7dff0207f6bd9c24ba`。
运行状态 `completed`；独立接纳 `True`；开发后继条件 `True`。

只使用旧开发72训练/16验证样本；R1资格32样本未载入。不是新独立资格或部署收益。
唯一新干预I+(h(a)-h(0))，幅度1；delta来自旧已接受归档，未训练或执行预测器。
oracle为同当前state/language/noise的未来视觉冻结策略输出，不是专家。

## train（episode等权）

| 条件 | 样本/episode | 首动作MSE | chunk MSE | token MSE |
|---|---:|---:|---:|---:|
| identity | 72/18 | 0.075968549259 | 0.088663766316 | 2312.744809170 |
| base | 72/18 | 0.064137491930 | 0.081749438083 | 2313.558247075 |
| centered | 72/18 | 0.055629032020 | 0.079294967078 | 2315.185317077 |
| identity_true | 72/18 | 0.053600007189 | 0.082743250036 | 2314.359247736 |
| identity_zero | 72/18 | 0.075968549259 | 0.088663766316 | 2312.744809170 |
| identity_mismatched | 69/18 | 0.117012616613 | 0.094466661634 | 2324.405550076 |

差值均为对照−干预，正数代表改善；错配采用相同子集。

base_vs_identity: N=72, episodes=18, 平均收益=0.011831057329, 样本方向={'worsened': 25, 'improved': 47}, episode改善=14。
centered_vs_base: N=72, episodes=18, 平均收益=0.008508459910, 样本方向={'worsened': 34, 'improved': 38}, episode改善=14。
identity_true_vs_identity: N=72, episodes=18, 平均收益=0.022368542069, 样本方向={'worsened': 22, 'improved': 50}, episode改善=14。
identity_true_vs_centered: N=72, episodes=18, 平均收益=0.002029024830, 样本方向={'improved': 32, 'worsened': 40}, episode改善=7。
identity_true_vs_identity_mismatched: N=69, episodes=18, 平均收益=0.059756622511, 样本方向={'improved': 53, 'worsened': 16}, episode改善=16。

新true相对I最不利样本（不删例）：

`[4, 46, 5]`: -0.642751500463 (worsened)
`[2, 49, 5]`: -0.039723351806 (worsened)
`[4, 48, 5]`: -0.030003770453 (worsened)
`[1, 46, 5]`: -0.016301291337 (worsened)
`[1, 46, 6]`: -0.008298329884 (worsened)

留一episode收益: `{"0/46": 0.022369080891204995, "0/48": 0.023597515971458954, "0/49": 0.023476340416168626, "1/46": 0.023482533652747488, "1/48": 0.022684723207957662, "1/49": 0.013805221062650955, "2/46": 0.02329687473081366, "2/48": 0.023685131227619328, "2/49": 0.023962990359847735, "3/46": 0.023418024984714594, "3/48": 0.019536105691252418, "3/49": 0.023450102020528715, "4/46": 0.03234946494811849, "4/48": 0.023913135425160794, "4/49": 0.023302637900272927, "5/46": 0.021952785808622987, "5/48": 0.011865925940811166, "5/49": 0.022485163009506436}`

## validation（episode等权）

| 条件 | 样本/episode | 首动作MSE | chunk MSE | token MSE |
|---|---:|---:|---:|---:|
| identity | 16/4 | 0.031709702433 | 0.053248021229 | 2670.828930525 |
| base | 16/4 | 0.028710898469 | 0.039825450841 | 2671.641088964 |
| centered | 16/4 | 0.013862918700 | 0.021462120248 | 2673.353070642 |
| identity_true | 16/4 | 0.012048372598 | 0.026728824581 | 2672.575387909 |
| identity_zero | 16/4 | 0.031709702433 | 0.053248021229 | 2670.828930525 |
| identity_mismatched | 16/4 | 0.044078063496 | 0.065970265658 | 2674.098964123 |

差值均为对照−干预，正数代表改善；错配采用相同子集。

base_vs_identity: N=16, episodes=4, 平均收益=0.002998803964, 样本方向={'improved': 10, 'worsened': 6}, episode改善=3。
centered_vs_base: N=16, episodes=4, 平均收益=0.014847979769, 样本方向={'worsened': 8, 'improved': 8}, episode改善=4。
identity_true_vs_identity: N=16, episodes=4, 平均收益=0.019661329835, 样本方向={'improved': 12, 'worsened': 4}, episode改善=4。
identity_true_vs_centered: N=16, episodes=4, 平均收益=0.001814546101, 样本方向={'worsened': 9, 'improved': 7}, episode改善=3。
identity_true_vs_identity_mismatched: N=16, episodes=4, 平均收益=0.032029690898, 样本方向={'worsened': 10, 'improved': 6}, episode改善=4。

新true相对I最不利样本（不删例）：

`[6, 48, 6]`: -0.039780288237 (worsened)
`[7, 48, 5]`: -0.009608695968 (worsened)
`[7, 49, 6]`: -0.002956454410 (worsened)
`[7, 48, 6]`: -0.000069679639 (worsened)
`[6, 49, 6]`: 0.000833158005 (improved)

留一episode收益: `{"6/48": 0.023052098219479195, "6/49": 0.022864730205533357, "7/48": 0.024922568334184148, "7/49": 0.0078059225798626205}`

## 判据与执行

```json
{
  "checks": {
    "better_identity": true,
    "better_centered": true,
    "better_identity_mismatched": true,
    "three_of_four_episodes": true,
    "strict_sample_majority": true,
    "chunk_not_worse_identity": true
  },
  "counts": {
    "vla_loads": 1,
    "decoder": 525,
    "identity_exact": 88,
    "base_exact": 88,
    "centered_exact": 88,
    "zero_exact": 88,
    "captures": 8
  },
  "execution": {
    "child_pid": 605582,
    "child_exit_code": 0,
    "exit_confirmed": true,
    "started_at_utc": "2026-09-16T23:11:47.478042+00:00",
    "finished_at_utc": "2026-09-16T23:12:35.601446+00:00",
    "wall_seconds": 48.12344381600269,
    "stop_reason": null,
    "forced_termination": false,
    "pending": {},
    "active": {}
  },
  "phases": {
    "started": 616,
    "returned": 616
  }
}
```

全部七维输出误差的交叉项/扰动能量分解保存于independent_audit.json；它不是视觉Jacobian或物理因果识别。
本轮不搜索alpha、不自动训练或重读资格集。旧R1阴性结论不变；闭环/时延/安全资格均未建立。
