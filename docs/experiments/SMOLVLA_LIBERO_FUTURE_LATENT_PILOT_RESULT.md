# F-LAT1：首个当前token回放差异，训练前停止

2026-09-10。execution HEAD `c5950d5195c83e30105563e7f8ffb646ecd7ebda`；登记5612079156已按实际ID单次exact回读。
原始输出 `outputs/smolvla_libero_future_latent_c5950d51/`，保持不变，不能重启。

原生承诺前缀的CPU对齐通过76对，train51/validation13/test12；11项CPU测试、模型入口、最终Ruff/format均通过。
唯一正式attempt加载原strict模型成功，missing/unexpected/shape mismatch为空。
task0/request3的第一批当前双相机token重新编码之后，原exact门抛出：
`ValueError: Current-token extraction differs from archive`。
运行仅发生1次双相机编码；尚无future编码、训练更新、predictor/decoder/Graph capture或新Env/native。
不是预测质量负结果，也不是GPU超时。原报告未保存该差异幅度，不能事后声称误差有多小。

child2863111 exit2、supervisor2863064 exit2均确认；监督8.119385s，独立外层11.576633s，
UTC03:12:50.775905–03:13:02.352515。stop_reason=null、active phase为空，没有强杀或未知挂起调用。
本轮offline_contract_passed=false；旧E-NAT1等结论及全部科学资格不变。

## 定位与独立修正版

源码已确认路径差异：本轮使用`policy_observation(raw)`先在CPU处理图像，原native worker使用
`build_dataset_frame → prepare_observation_for_inference(..., cuda) → preprocessor`，uint8归一化在目标设备执行。
这确认了复现入口不一致，尚未以本轮原件隔离证明特定GPU算子是数值差异的唯一原因。
F-LAT1-r1只改为保存的worker_observation及同一native准备路径；不放宽token或动作exact门，不改模型/数据划分/训练。
原F-LAT1终态保留；r1须独立冻结登记和输出，不能在旧目录修后重跑。
