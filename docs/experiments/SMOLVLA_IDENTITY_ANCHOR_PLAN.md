# F-IAR1：固定残差的基底置换，原开发集机制诊断

2026-09-17。用户在收到F-ACQ1-R1有效阴性结果后要求继续。本轮不是重跑资格实验，也不追加资格样本。

## 问题和唯一干预

R1双门未过，不能部署。回原训练72例/18episode及反复使用的开发验证16例/4episode，区分B和中心化残差的影响。
保留F-ACR1第72步已保存的同一h(a)、h(0)，唯一新干预为把 `B+(h(a)-h(0))` 改为 `I+(h(a)-h(0))`。
I是同一样本的当前原始视觉token，FP32加法、先减后加，最后BF16往返。幅度固定1，不搜索系数、不训练/选点。
这是冻结残差数组移植，不是重新训练的identity-centered模型，也不是完整预测器部署计算时延。

允许数据只有F-ACT1/F-COV1原开发来源、F-PFX1无动作基底归档、F-ACR1所选centered归档及来源审计/权重哈希。
F-ACQ1-R1的32样本、新初态、task8/9标签、confirmation完全不载入；原pending/R1文档只做保留哈希，不用于分析。
旧失败/R1输出不改写，不宣称新任务/独立test或新闭环收益。结果始终是开发证据。

## 固定次序、对照和预算

沿用原72/16 split、88个key、mask、3个delay4单例，按task/state/request排序，不选好例。
每例先重新decode I、B、原B+delta，与各自已接受归档完整50x32逐值exact；首不一致就停止。
随后只decode I+delta_true、I+delta_zero及有供体的I+delta_mismatched；同split/task/state/delay循环下一request，单例无错配。
训练错配69例、验证16例；错配比较强制同一85例接收者子集。零动作raw token和完整输出均必须exact回I。
全部新token、完整输出、供体、每样本指标保存，不把B的缓存成本说成部署开销。

正式decoder恰好525 = 88x5 + 85，其中264旧控制兼容重放、88true、88zero、85mismatch。
VLA加载1；predictor模型加载/前向0（只读已保存delta）；训练/反传/新编码/新Env/真机均0。
Graph captures<=8（按task排序、每task同一语言形状）；内部setup/warmup/capture每次1/3/1另记。
加载开发数据<=30秒、VLA<=90秒、全部配对推理<=180秒、单例<=60秒、单decoder<=30秒。
总soft300/hard330秒，工具总预算390秒；attempt1/retry0，首错保存，不修后重跑正式worker。
仅管理自有进程组，TERM等待5秒后才KILL；前台监督、同一Job跟踪到退出，保存phase与退出证据。
复用已核验的专用Python/uv/offline/EGL/LIBERO_CONFIG_PATH/LD_PRELOAD设置，删除PYTHONPATH，不安装依赖。

## 预先定义的分析

CPU float64，首动作归一化7D MSE为主，同时50x7D chunk、有效视觉token MSE；episode等权。
数值rtol1e-6/atol1e-7，方向阈值1e-7+1e-6*abs(control)，exact绝不放宽。
固定对比：B vs I、原B+delta vs B、新I+delta vs I、vs 原B+delta、vs 新I+错配delta。
完整报告训练和开发验证、逐episode/样本方向、留一episode、最坏与最好样本、有效配对分母。
附加纯描述性动作输出误差分解：delta为两组完整输出之差，验证MSE变化=2*mean((control-oracle)*delta)+mean(delta^2)。
这不是视觉模型Jacobian或真实物理因果解释，不给七维索引擅自附加控制器语义。

`development_followup_supported`只指下一步开发是否有依据：新I+true在开发验证首动作macro严格优于I、原B+delta及同组错配，
相对I至少3/4episode和>8/16样本改善，chunk不劣于I；全部同时满足。不称资格通过/显著性/安全门。
无论结果方向如何都停止本轮，不自动训练、扫alpha、读取R1新样本、扩样或部署。

## 准备、发布、审计

新脚本/测试/本计划先显式提交推送；保留13份原未提交文件字节，不git add全部。
CPU prepare绑定实际HEAD、source SHA256、原split/供体、环境及旧worktree哈希；唯一按HEAD的PREP/OUT，不复用旧目录。
先GET新唯一标识F-IAR1-REGISTER:<HEAD>，不存在才独立gh literal argv POST一次，然后按实际ID独立GET。
正文与本地exact，保存实际POST/GET的规范化投影及来源说明，才允许运行。POST状态不明不重发。
权限/安全拒绝停止对应动作，不更换通道绕过；普通参数或搜索后端问题可透明最小修正。
退出后单次独立CPU审计，使用NumPy重新归约所有保存预测并核对原始来源/组合/供体/预算/phase闭合。
分析代码与审计会共享固定数据身份装载，指标归约和推进判据必须独立实现。
只保存新报告/JSON/审计/回执；原资格结果不会被本轮开发结果覆盖。成功的开发发现也不改变R1阴性结论。
延续发布限制：必要预登记允许；开发推进条件未满足时仅本地保存并向用户如实报告，不发布结果评论。
生产默认、baseline_qualified/realtime_qualified/predictor_benefit_tested均false，risk_thresholds=null。
