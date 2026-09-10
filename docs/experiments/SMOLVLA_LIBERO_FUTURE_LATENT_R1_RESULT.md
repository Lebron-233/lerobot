# F-LAT1-r1：未来视觉预测完成，动作条件优势尚未建立

2026-09-10。执行HEAD `63f1f3064b4aae954e63a0149f85a7ca221afc3a`，预登记5612120361按实际ID单次GET正文exact后运行。
原件 `outputs/smolvla_libero_future_latent_r1_63f1f306/`；本轮唯一attempt，训练/test完成后未重训或追加样本。

## 结论

原生worker准备路径、实际承诺前缀→未来观测对齐、现有轻量预测器训练和同noise的Graph动作比较已经贯通，
运行及独立CPU核验通过。**但动作条件创新尚无支持性对照结论**：conditioned的token MSE略优于no_action，
而no_action在row0与完整chunk到oracle视觉输出的距离上更好。不能把任一相对identity改善直接归因于动作输入。
这是固定开发数据上的小规模离线pilot，不是闭环成功率提升。

## 数据与冻结训练

只使用E-NAT1的10条async轨迹，76个真实planned前缀；全部delay3。task1/request8因结束前缺完整future被显式排除。
train任务0–5共51对，validation6–7共13对，test8–9共12对。任务/轨迹不跨集合；没有新增Env或native step。
当前原生两相机scaled token、model-ready32Dstate及normalized7D承诺动作输入；未来图像仅监督/reference。
真实post7D前缀只用于来源核对，不积分为未来state。所有比较保持同一当前state、language与归档noise。

复用69,680参数的LightweightFutureLatentPredictor，rank16、risk_head关闭。
conditioned使用真实动作；no_action仅将有效动作置零，其余结构/初始化/采样种子/预算相同。
两臂各200updates，AdamW lr0.001、wd0.0001、batch8、clip1；VLA/视觉编码器冻结。
验证在step0及每25更新进行，按native-bf16 token MSE选择最早最低点（包含identity初始解）。
conditioned选择step125，validation MSE2163.696044922；no_action选择step175，validation MSE2163.007080078。
两检查点固定后各评test一次，保存预测token及全部动作输出，没有根据test调参。

## 留出集结果：12对、2个任务

误差越低越好；MSE以原scaled token或policy-normalized有效7D计算。

| 指标 | identity | conditioned | no_action |
|---|---:|---:|---:|
| future token MSE | 2033.826955160 | 2014.205235799 | 2024.928888957 |
| token误差相对identity降低 | — | 0.964768% | 0.437504% |
| row0到oracle视觉输出MSE | 0.028693651509 | 0.026501742458 | 0.024484635362 |
| row0误差相对identity降低 | — | 7.639004% | 14.668806% |
| 50行chunk到oracle视觉输出MSE | 0.038285209065 | 0.032709646133 | 0.031701493766 |
| chunk误差相对identity降低 | — | 14.563230% | 17.196498% |

两任务各6对，sample mean与task macro只差浮点舍入。

| 留出任务 | identity token MSE | conditioned | no_action |
|---|---:|---:|---:|
| task8 | 2288.263610840 | 2249.355773926 | 2245.342488607 |
| task9 | 1779.390299479 | 1779.054697673 | 1804.515289307 |

完整逐样本值及逐任务oracle-action指标保存在原result.json；机器摘要记录主指标和核验结果。
oracle在此仅指“真实未来视觉token＋当前state”进入同一策略产生的输出，不是专家/最优动作或success上界。
token与动作指标排序相反，说明此pilot中不能用latent MSE排名替代动作层判据。

## 运行、核验与资源

86次双相机编码=76未来+10当前复现；两臂共400updates；test12×4=48正式decoder。
每留出任务1次Graph capture，共2；原setup/warmup/capture内部2/6/2另记，不混入正式decoder数。
模型只加载一次，missing/unexpected/shape mismatch均空。新Env/native/真实机器人为0。
10个task的当前两相机token最大绝对差均0，state均exact；12个identity完整50x32输出与原native归档exact。
两个checkpoint/9点验证历史的选择复核一致，69,680参数，risk关闭。

退出后只用CPU读取本轮保存的cache/预测token/动作，没有forward、训练或新增评估样本。
108个逐样本指标以CPU float64独立复算，对原GPU float32最大相对差1.736647448e-7，
在rtol1e-6/atol1e-7的归约核验容差内。此容差不用于token/动作输入复现，后者始终逐值exact。
139组phase起止完整，无错误或开放阶段；86编码/48decoder/2训练阶段与原计数对应，全部时限/预算满足。
Python/version/140包metadata前后一致。独立核验0.054611s，CUDA未初始化。

child2864128 exit0、supervisor2864081 exit0，退出均确认；first_failure/stop_reason为空，无TERM/KILL。
监督wall15.734335563s、独立外层19.270006976s；UTC03:18:23.985946–03:18:43.255929。
这些是离线管线时间，不是预测器单次时延或实时资格评估。

## 原F-LAT1首错与修正

原execution c5950d51在第一批当前token编码后exact不一致，训练0/decoder0，child和supervisor均exit2；原件和失败报告保持。
r1仅改为归档worker_observation与原native图像准备路径，在原设备完成uint8归一化，而不是CPU raw重建入口。
原exact门没有放宽，修正后全部10个当前案例通过。旧差异幅度未保存，不补称已测或把某一个算子当唯一已证实原因。
原11项CPU通过，新路径2项通过（11 deselected），两次准备及所有原始失败日志保留。

## 限制与下一步裁决

76对来自10条已知开发轨迹、单一state41及delay3，不能证明不同延迟/新初态/长轨迹泛化。
不训练state或校准risk，未来视觉oracle只在离线使用。conditioned与no_action动作距离排名的差异未做统计显著性结论。
旧E-NAT1等历史字段保持；baseline_qualified/realtime_qualified/predictor_benefit_tested仍false，risk_thresholds=null，旧confirmation untouched。
下一阶段须以新的训练/验证/留出数据复核动作输入的增量价值，不在本轮12例上调参；本轮不直接进入在线部署或开启生产默认。
当前predictive_async在线predicted路径仍有固定6D动作准备，原生7D接入需单独定义，不能直接把本实验checkpoint当可部署格式。
