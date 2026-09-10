# F-SCL1接续：先核对三个delay4来源，不能跳过门重跑

F-SCL1执行077ce452已在select_data终止，模型/训练/解码/native均0。
原件 `outputs/smolvla_case_scale_077ce452/` 保留为终态，报告见SMOLVLA_CASE_SCALE_RESULT.md/json。

已确认F-COV1 new_labels.pt含45个train delay3、3个train delay4、16个validation delay3。
三例为task3/state48/request6、task3/state49/request6、task4/state48/request6。
旧F-COV1“64例全部delay3”文案不正确；本轮已另记勘误，不改旧原件和数值。

最小接续范围：
1. 在正常获授权的执行环境核对上述三例原计划、observations绝对索引、保存prefix/mask和native来源。
   原目录分别episode_003、episode_012、episode_004，只读所需三例，不重新扫描整个旧实验。
   本会话该查询被平台拦截，未取得结果，不能写成已通过。
2. 若原证据一致，保留全部72train/16validation，按实际delay使用三例4，其余3，不删例、不重写标签。
   新CPU准备直接核对实际数据；新增允许真实混合延迟及错误delay/mask负例测试。
   保持四组同数据/顺序/72更新、原架构/优化器/损失定义和已固定floor规则，不趁修正调参。
3. 保存原失败，独立F-SCL1-r1命名空间/源提交/预登记/独占输出，准备与登记exact通过后只运行一次。
   模型计算前所有真实数据门必须通过；旧F-SCL1不是可恢复队列。

不得用任何已报告test补样/选点；不加载旧选定预测器继续训练；不扩大损失系数扫描或开启在线部署。
当前没有修正版运行，也没有后台实验。旧待提交文档保持。

正式停止报告结果提交393d84f3已推送，评论5615594200已按实际ID单次GET正文exact；发布回执另行保存。
