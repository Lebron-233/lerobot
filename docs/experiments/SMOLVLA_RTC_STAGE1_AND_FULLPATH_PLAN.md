# RTC三阶段接续：阶段一修复与阶段二固定全路径测量

2026-09-17。接续c020b2e8/Issue #1评论5707570138，用户明确要求执行三阶段。
本计划不重训视觉预测器、不读取IQ1/ACQ1资格标签，不升级依赖/驱动，不改变生产默认。

## 阶段一：实际首错和最小修复

在原冻结denoiser/no_grad条件下，RTCProcessor先前在denoiser前向之后才对x_t开启requires_grad。
解析公式为x_clean=x-t*v(x)，correction=(I-t*J_v)^T*weighted_error。原路径丢失J_v。
合成测试先真实失败5项/通过2项，最大绝对差6.0；移动requires_grad到前向之前后这5项通过。
新增7D到32D padding测试又真实失败：25个非动作维度被当成观测为0的目标。现对误差做有效维度掩码，并保留输入dtype。
两份旧固定期望测试原先也按identity Jacobian计算；用解析系数1-t*0.5=0.75修正期望，不放宽容差。
最终RTC原回归+新增解析/时间身份测试共261通过。Junit保存于outputs/smolvla_rtc_stage1_development_20260917/tests.xml。

新rtc_execution_queue.py在原ActionQueue之上提供原子请求快照：观测索引=下一实际动作索引；原始归一化7D与后处理7D分别保存；新块row i对应观测索引+i。
完成时只丢弃一次实际已消费行，使用动作索引差而非墙钟推算；旧epoch/旧请求、关闭和超出cap8返回均不安装。
测试delay0/1/3/4/8、delay>=chunk、预测和实际消费不一致、前缀不足、无动作不推进索引、reset及待处理线程关闭。
它是待集成工程接口，不把CPU测试当成闭环完成，不继续沿用未来token方案的未来时刻row0约定。

主要公式来源（仅查证，不下载执行）：
https://raw.githubusercontent.com/Physical-Intelligence/real-time-chunking-kinetix/main/src/model.py （realtime_action完整jax.vjp）
https://huggingface.co/docs/lerobot/rtc （RTC引导/配置边界）

## 阶段二：RTC-FULLPATH1固定合同

仅原F-COV1 task6/7 state48/49 request3..6的16例；直接读取原四条开发episode的当前原始双相机/状态。
前缀从该时刻实际在执行的旧请求policy_chunk的row20起取30x7原始归一化tail，与原承诺前三行exact核对；不误用当前请求输出作前缀，不用后处理动作，不读future/oracle选参。
同一个冻结SmolVLA，chunk50、解码10、20Hz；无RTC与guided RTC同eager路径，无Graph或预测器。
RTC配置固定EXP、execution_horizon10、max_guidance_weight10、mode=guided；不使用trained模式。
每样本每臂6次，共192请求。repeat0两臂均无前缀以核对公开API与原完整50x32输出exact，其余5次RTC有前缀；按样本及repeat奇偶交替两臂。
每臂80次后续测量，首次真正的guided调用也保留其中，没有隐藏预热/补测或挑低负载窗口。不是严格隔离的冷启动benchmark。

计时从原始观测转换/预处理、输入搬运到policy.predict_action_chunk、后处理及CPU输出完成，CUDA同步封口。
通过既有profiled接口记录vision_encode、prefix_prefill、flow_matching；整路径含观察用CPU证据复制，两臂一致。磁盘写入在计时外。
queue_wait_s=null，离线测量不是线上排队或环境重叠；不把public_predict_total和其内部子阶段相加重复计时。
每次实际重新编码图像；192次输入token/mask/state/语言与旧记录exact，112次无引导完整输出exact，16无前缀对照exact。
预期1次VLA加载、192完整请求和视觉编码、960 RTC包装步，其中800有前缀VJP；训练/Env/资格标签/Graph0。

全路径P50/P95/P99按nearest-rank，不按样本先平均来隐藏尾延迟。进入三臂仿真门：两臂均ceil(P99秒*20)+1<=8，沿用20Hz/cap8及一行动作余量，且所有数值/输入/退出审计通过。
若尾延迟超限，只记录当前eager实现不满足本预算；不据此声称RTC普遍不可实时，不降频/扩大cap/改引导强度来通过。
无前缀或模型技术首错停止该worker并封存，不重启或静默变更192预算。每请求30秒，模型90秒，soft600/hard630，工具690秒，attempt1/retry0。

## 登记、审计及阶段三边界

代码测试通过后显式提交/正常推送原分支。CPU prepare绑定HEAD、来源/源码/权重哈希、专用环境和21旧pending原字节，生成唯一PREP/OUT。
预登记RTC-FULLPATH1-REGISTER:<HEAD>，先gh literal argv只读查重，单独POST一次，实际ID独立GET并body exact落地。
监督一个自有worker，TERM后最多等5秒才KILL，日志/退出/phase闭合；明确权限安全拒绝停止对应动作，不换通道绕过。
退出后单次CPU独立复算192记录、无前缀/基线exact、nearest-rank分位数与尾延迟判据；不重新执行模型或VJP。
只有阶段二技术与时延门均通过才冻结24episode三臂新合同；否则不启动阶段三。
完整解析修复或时延边界属于有意义工程发现，可在Issue #1报告，必须保留全部不利证据和未完成阶段，不宣称部署、闭环成功或统计泛化。
