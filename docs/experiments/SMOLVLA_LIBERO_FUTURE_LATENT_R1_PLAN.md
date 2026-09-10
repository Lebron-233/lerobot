# F-LAT1-r1：原生worker图像准备路径的离线预测试验

原F-LAT1执行c5950d51因首个当前token与归档不一致终止，训练0、decoder0；原失败报告及输出保持。
依据用户当前继续实验指令，本次独立修正版仅纠正新离线入口，不替换旧尝试或改变旧E-NAT1。

继承F-LAT1固定PLAN的任务划分、模型、损失、两臂、种子、200更新、验证选择、test一次和全部资源预算。
唯一计算入口差异：使用归档worker_observation重建原features，调用原build_dataset_frame和
prepare_observation_for_inference(..., cuda, task, "libero")，再使用同一preprocessor。
uint8转浮点/255/permute在原GPU位置完成；不重复raw像素旋转或另造观测。
current两camera token和state仍必须逐值exact，每task首样本数值差异持久化；任一差异首错停，不用allclose替代。

复用已通过CPU对齐的同一76样本缓存，不重复扫描原3GB数组；train51/validation13/test12，全部delay3。
仍为视觉预测、当前32Dstate保持；真实normalized7D动作与no_action消融、risk关闭。
上限仍87编码批次、400更新、96正式decoder、2capture及内部2/6/2；计划实际为86编码及48decoder。
新公开输出前缀`outputs/smolvla_libero_future_latent_r1_<HEAD前8位>`，新准备目录
`outputs/smolvla_libero_future_latent_r1_preparation_c5950d51`。
attempt1/retry0，技术首错不改后重跑；无新增Env/native/真机。原冻结训练/test规则和不可提升资格边界完整继承。

旧11项CPU/76数据对齐已通过；仅新增2项测试检查worker原features、保留uint8输入和不重复raw旋转。
实际入口/环境/lint准备完成并推送、GitHub登记exact回读后才唯一执行。
如果本轮消除了current-token差异，只能接纳新路径下的复现，不将先前未知差异幅度补写为已测。
