# F-ITC1-R1：一次离线指标设备边界技术恢复

2026-09-17。用户要求继续实验；原F-ITC1在任何训练更新前发生代码设备错误，详见
SMOLVLA_IDENTITY_TRAINING_INITIAL_FAILURE.md。该失败及其attempt1/retry0、预登记5706253063封存。
恢复另外绑定新execution HEAD、新PREP/OUT及真实预登记。ITC系列派发后总尝试为2，
恢复执行1，执行内重试0；旧VLA1/预测器2/predictor2/decoder3/capture1须额外报告。
本增补只许可一次技术恢复；新执行遇首错继续封存，不再启动第三次。

仅新增offline_metrics：将离线visual和value detach到CPU再调用原float64指标归约；
初始重放和最终评估都使用这个边界。梯度训练仍使用q.cov.metric_values，梯度路径不detach。
不改变任何样本/供体、模型/初始化、72次更新、学习率、目标函数、系数、科学判据或容差。
原SMOLVLA_IDENTITY_TRAINING_PLAN.md全部科学条款沿用；修复不使用训练或验证结果调参。

修复提交前执行CPU测试与Ruff，并仅用合成CUDA token、CPU参照和合成动作输出测试
离线指标函数，模型加载/前向/Env/科学样本0。该诊断确会初始化CUDA，必须与CPU准备分进程，
不冒称CPU-only，也不占正式模型调用。原固定环境、解释器、驱动/依赖均不更换。

提交推送后，对新HEAD执行一次CPU prepare；不调用旧prepare。预登记同时包含
F-ITC1-R1-REGISTER:<新HEAD>与兼容F-ITC1-REGISTER:<新HEAD>、原失败SHA、完整输出、
新preparation SHA256、完整原科学合同、恢复范围、原消耗和系列累计次数。
先查唯一标识，再用gh literal argv单独POST一次、真实ID独立GET、正文exact落地。
单次监督入口运行到终态，退出后一次独立CPU审计，失败也保留真实报告。
原预算每执行decoder930/predictor1508/更新与反传各144/VLA1/预测器2；
soft900/hard930秒、工具990秒，capture<=24且内部调用另计。没有额外训练机会。

原失败及恢复都永久保留。不根据恢复结果再修改源文件来获得接纳，不覆写审计或放宽容差。
成功条件仍仅原guarded固定全门，不改选plain；条件未通过则只本地报告，阳性才发布结果评论。
明确安全/权限拒绝停止对应动作，不换工具绕过；普通设备错误可在新登记的技术恢复中最小修正。
R1资格集32样本仍不读取；无新Env、新图像编码、真机、时延、闭环或第三次运行。
