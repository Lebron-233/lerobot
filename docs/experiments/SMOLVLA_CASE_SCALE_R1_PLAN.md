# F-SCL1-r1：真实混合延迟的数据门修正，尺度对照保持

接续 eae8c863 / Issue #1 5615594200。原077ce452只在CPU数据门失败，
独占输出与旧源码提交保持终态；r1是新源提交、新登记、新输出的一次运行，不是续跑。

## CPU原生来源核对（2026-09-10）

只读F-COV1三个episode的request6及其必需数组，不重审其他episode：
episode003=task3/state48、episode012=task3/state49、episode004=task4/state48。
三例原plan均next_action_index89/takeover_index93/planned_delay_steps4。
实际native动作索引89–92，均来自request5的第20–23行；原normalized前缀、
post前缀、实际native、标签动作/mask、当前和未来索引及cached inputs逐值一致。
最后一个前缀native返回早于future观测，future观测早于下一条实际动作。
证据在新preparation的source_trace.json，CPU exit0，无forward/native。

旧F-COV1“64例全部delay3”文案应为train45×delay3+3×delay4、validation16×delay3。
本核对补全三例来源，不改旧数值、不重复训练F-COV1、不重新接纳整个旧实验。

## r1唯一修改

保留原[尺度计划](SMOLVLA_CASE_SCALE_PLAN.md)全部数据、四组、采样、损失/权重、
floor=max(0.1*Sa,1e-12)、均值1权重、train25%低误差阈值和判读规则。
数据门不再错误地要求所有delay3，而是精确验证三个指定key为4，其余为3；
future-current=delay、mask为真实delay的连续前缀、动作[1,8,7]有限且padding零。
共同train72=69×delay3+3×delay4；共同validation16均delay3，全部样本原样保留。
不把3个4改成3、删例、换未来观测，也不接受其他key的任意混合延迟。
启动前CPU准备直接加载两份实际开发标签，执行完整select_data并保存源hash、
身份/delay摘要与原生source_trace；worker在模型加载前重核这些小数据门。

global_conditioned/case_conditioned/global_no_action/case_no_action各69680参数，
同seed20260912，72更新/batch1，同原multi顺序，每task/state四次，每样本一次。
AdamW lr0.001/wd0.0001/clip1；Sz/Sa仍来自原47train，token项与当前32Dstate不变。
0/36/72用同一validation原始row0 episode均值选最早最小值，包含0步；
报告共同train/validation、低误差子群与所有消融，不把返回identity当学习收益。
这是已使用过的开发验证，不是独立测试；任何task8/9测试及confirmation21–40不读。

## 预算、停止与交付

模型加载1；正式完整decoder812=44参考+192验证+288可导训练+288所选train评估；
总上限900、反传/更新各288，Graph capture上限64及内部setup/warmup/capture另记。
新增Env/native/图像编码/test/真机/VLA训练均0。每次解码/更新30s、每组600s，
外层1400s TERM/1430s KILL、自有进程TERM后5s未退才KILL；attempt1/retry0。
原uv offline/no-project/no-download、固定Python/140包；不安装或干预同卡进程。
CPU新增正负例与原10项测试，静态检查和无CUDA入口通过，再提交推送、登记exact后执行。
任何首错停止，不扩样、不回读test调参。退出只用CPU保存数组复算尺度、288步顺序、
验证选点和所有数值/低误差子群/预算/退出；输入动作exact不放宽，归约rtol1e-6/atol1e-7。
结果和回执分开提交，旧F-ACT1待提交三文件保持；生产默认/闭环资格false、risk_thresholds=null。
