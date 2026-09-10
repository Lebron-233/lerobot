# F-OPT1：训练／验证范围的同样本优化诊断

接续 F-ACT1 / 评论5614106958，依据本轮用户继续实验指令。
问题：有梯度但联合监督回退identity，究竟能否降低刚用于更新的那个样本的动作误差？
本轮只提供可学习性、步长和样本间迁移诊断，不是新的测试成绩或闭环收益。

## 数据和四个预设设置

只读 `outputs/smolvla_action_objective_1666c067/development_labels.pt`：原47训练／12验证。
不读取test_cache、test_outputs或任何旧测试指标，不启动Env、视觉编码或重采标签。
按每个训练task0–5最小request_id各取一个anchor，合计6个；不按误差/成功挑选。
其余41个训练样本仅用于复用原训练尺度Sz、Sa，不参与本轮梯度。
task6–7的12例继续明确称为已使用过的开发验证集，不当作盲测。

四组独立同seed20260912、同69680参数零残差初始化，各24步，batch1，六anchor顺序循环四遍。
固定比较：joint_original（Lz/Sz+La/Sa，lr1e-3）；joint_small（同目标，lr1e-4）；
action_small（仅La/Sa，lr1e-4）；joint_small_no_action（joint/lr1e-4，动作置零）。
AdamW wd1e-4、clip1，原bf16转换及完整十步sampler保持；其他设置无扫描或结果后修改。
Lz是量化后future token MSE，La是原有效7D row0与已存oracle视觉参考输出MSE。
state始终当前model-ready state；语言、采样noise和teacher对每次前后评估完全相同。

## 记录和解释

每次更新实际保存更新前后的两camera token和完整50×32输出，独立复算La/Lz与变动元素数。
非零梯度不自动视为误差降低；报告下降/相同/上升次数、每anchor首末和干扰。
0/24步对6训练anchor+12验证例评估，分别按episode等权汇总。
训练下降而验证未降只能说明开发范围泛化不足；单次局部上升不能唯一归因于bf16。
四组为定向开发比较而非统计超参搜索，不据负结果追加步数、anchor、seed或新测试。
即使非零检查点验证胜出，也只是后续新数据验证候选，科学资格不提升。

## 预算、执行与交付

完整decoder336=96可导训练+96同样本更新后+144个0/24评估；backward96、updates96。
Graph capture0、视觉编码0、Env/native/test读取0；不使用Graph反传，不增加reference调用。
训练阶段VLA冻结且所有梯度None；仅predictor更新。
每update或assessment30s、每arm300s，外层1200s TERM/1230s硬上限，自有进程5s未退出才KILL。
一次独立attempt，首错停止，不重试、不将旧F-ACT1重新启动。

先提交本轮新代码/测试/PLAN并登记完整HEAD、命令、独占output；评论按实际ID回读exact。
旧F-ACT1待提交的三个回执/交接文件保留原样，不纳入本轮源提交；允许的工作树例外仅为这些文档及原三个未跟踪文档。
旧实验source gate不修改；新入口仅校验本轮HEAD和明示文档例外，任何其他源码变动拒绝。
执行前后比较原模型Python/包metadata，不安装升级、不操作其他GPU进程。
退出后CPU核验数值、采样顺序、冻结源标签、初始identity及非零残差、全部phase和退出，不能用核验重跑forward。
新结果和接续建议另存独立文档，不覆盖旧报告或待提交回执。
