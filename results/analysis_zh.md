# AG News 实验结果

|系统|正确数/总数|Accuracy|95% Wilson 区间|Macro-F1|无效输出率|秒/条|
|---|---|---|---|---|---|---|
|qwen_extra_fixed|229/400|57.25%|52.36%–62.01%|0.5281|6.25%|0.102|
|qwen_extra_retrieval|190/400|47.50%|42.65%–52.39%|0.4442|1.25%|0.087|
|qwen_icl|229/400|57.25%|52.36%–62.01%|0.5281|6.25%|0.102|
|qwen_lora|352/400|88.00%|84.45%–90.83%|0.8790|0.00%|0.118|
|smol_extra_fixed|97/400|24.25%|20.31%–28.68%|0.1118|7.75%|0.189|
|smol_extra_retrieval|113/400|28.25%|24.06%–32.85%|0.1946|21.00%|0.277|
|smol_icl|97/400|24.25%|20.31%–28.68%|0.1118|7.75%|0.189|
|smol_lora|334/400|83.50%|79.55%–86.82%|0.8340|0.00%|0.154|

## 同一模型中 ICL 与 LoRA 的差异
- qwen: LoRA − ICL = +30.75 个百分点。
  训练准确率 88.25%，测试准确率 88.00%；差距 0.25 个百分点。结合验证 loss 曲线判断过拟合，不能仅凭训练 loss 下降下结论。
- smol: LoRA − ICL = +59.25 个百分点。
  训练准确率 82.00%，测试准确率 83.50%；差距 -1.50 个百分点。结合验证 loss 曲线判断过拟合，不能仅凭训练 loss 下降下结论。

## 配对错误分析

|系统A|系统B|A对B错|A错B对|两者都错|精确McNemar p|
|---|---|---|---|---|---|
|qwen_icl|qwen_lora|16|139|32|1.176e-25|
|qwen_icl|smol_icl|193|61|110|3.791e-17|
|qwen_icl|smol_lora|24|129|42|1.393e-18|
|qwen_lora|smol_icl|261|6|42|4.104e-69|
|qwen_lora|smol_lora|29|11|37|0.006427|
|smol_icl|smol_lora|9|246|57|3.904e-61|

配对检验是探索性分析，未进行多重比较校正；单次训练不能反映训练随机性。

## 错误示例

### qwen_icl
- test-1，真实：World；生成：'Business'。
  新闻：In China's changing society, psychotherapy gains acceptance BEIJING -- Step back, Confucius. Move over, Mao. Dr. Freud is making the rounds. Once vilified by the communists as a remnant of bourgeois imperialism, the practice of psychology -- and especially the Western ritual of going to a therapist -- is gaining popularity in this country of budding capitalists.
- test-8，真实：Sports；生成：'World'。
  新闻：We owe Athens an apology ATHENS -- The Games of the XXVIII Olympiad -- the great disaster that wasn #39;t -- come to an emotional end this afternoon and, really, the world owes Athens an apology.
- test-10，真实：World；生成：'Business'。
  新闻：New Rebel Factions Emerge in Darfur New rebel factions have emerged in western Sudan, complicating peace talks on the conflict in Darfur. The UN special representative for Darfur, Jan Pronk, says he thinks the new groups are serious and need to be taken into consideration.
- test-11，真实：World；生成：'Business'。
  新闻：At least 25 bodies at Sadr #39;s religious court NAJAF, Iraq : At least 25 charred and bloated bodies were discovered in the basement of a religious court set up by rebel cleric Moqtada Sadr in Najaf #39;s Old City, police said.
- test-13，真实：World；生成：'Business'。
  新闻：Philippine Rebels Free Troops, Talks in Doubt  PRESENTACION, Philippines (Reuters) - Philippine communist  rebels freed Wednesday two soldiers they had held as "prisoners  of war" for more than five months, saying they wanted to  rebuild confidence in peace talks with the government.

### qwen_lora
- test-25，真实：Sci/Tech；生成：'World'。
  新闻：UN 'must ignore cloning ban call' The UK's Royal Society urges the UN to ignore a call by President Bush to ban all forms of human cloning.
- test-45，真实：Sci/Tech；生成：'Business'。
  新闻：MedImmune CEO Talks Finance and Science David Mott, a Dartmouth-educated Wall Street investment banker, is increasingly leveraging his reputation in the local and national biotech communities.&lt;br&gt;&lt;FONT face="verdana,MS Sans Serif,arial,helvetica" size="-2" color="#666666"&gt;&lt;B&gt;-The Washington Post&lt;/b&gt;&lt;/font&gt;
- test-46，真实：Business；生成：'World'。
  新闻：Dark arts of spin evident in phoney war for Abbey THE phoney war over the fate of Abbey grinds on. Along the way, on all sides, it is producing its predictable crop of knowing winks and destabilising nudges.
- test-60，真实：Sci/Tech；生成：'Business'。
  新闻：Bourses set for losses after Wall St falls (FT.com) FT.com - European equity markets were poised for opening losses on Thursday following a weak session on Wall Street overnight, while caution was likely ahead of results from Nokia, the leading mobile handset maker.
- test-68，真实：Business；生成：'World'。
  新闻：Poor nations seek WTO textile aid With 40 years of textile quotas about to be abolished in a move to help developing nations, a group of the world #39;s poorest countries are asking for a different approach: special trade deals to protect them from a free-for-all.

### smol_icl
- test-2，真实：Sports；生成：'World'。
  新闻：Braves hope Furcal #39;s situation doesn #39;t tarnish race While Rafael Furcal #39;s DUI arrest on Friday could serve as a distraction for the remainder of the season, the Braves are looking to put the matter behind them, and at the 
- test-3，真实：Business；生成：'World'。
  新闻：Oil Falls Below \$49 on Nigeria Cease-Fire  LONDON (Reuters) - Oil prices dropped from record highs  above \$50 a barrel on Wednesday as the U.S. government reported  a surprise increase in crude stocks and rebels in Nigeria's  oil-rich delta region agreed a cease-fire.
- test-4，真实：Business；生成：'Return exactly one label: World, Sports, Business, or Sci/Tech.'。
  新闻：Hurricane loss 'less than feared' Hurricane Frances could cause \$3-6bn in insured losses in the US, less than experts first predicted.
- test-5，真实：Sports；生成：'World'。
  新闻：Bettman says  #39;season is likely slipping away #39; National Hockey League commissioner Gary Bettman doesn #39;t appear optimistic that the current player lockout will end soon, according to a televised report.
- test-6，真实：Sports；生成：'World'。
  新闻：Game On On the same day that Tiger Woods was married in Barbados, the two best golfers in the world were walking the streets of St. Andrews, Scotland.

### smol_lora
- test-1，真实：World；生成：'Business'。
  新闻：In China's changing society, psychotherapy gains acceptance BEIJING -- Step back, Confucius. Move over, Mao. Dr. Freud is making the rounds. Once vilified by the communists as a remnant of bourgeois imperialism, the practice of psychology -- and especially the Western ritual of going to a therapist -- is gaining popularity in this country of budding capitalists.
- test-3，真实：Business；生成：'World'。
  新闻：Oil Falls Below \$49 on Nigeria Cease-Fire  LONDON (Reuters) - Oil prices dropped from record highs  above \$50 a barrel on Wednesday as the U.S. government reported  a surprise increase in crude stocks and rebels in Nigeria's  oil-rich delta region agreed a cease-fire.
- test-4，真实：Business；生成：'World'。
  新闻：Hurricane loss 'less than feared' Hurricane Frances could cause \$3-6bn in insured losses in the US, less than experts first predicted.
- test-8，真实：Sports；生成：'World'。
  新闻：We owe Athens an apology ATHENS -- The Games of the XXVIII Olympiad -- the great disaster that wasn #39;t -- come to an emotional end this afternoon and, really, the world owes Athens an apology.
- test-16，真实：Business；生成：'World'。
  新闻：Treasury Chief Urges Debt Relief for Poor Nations Global lenders need to offer more grants and debt relief to poor countries and tailor lending toward the private sector, Treasury Secretary John Snow said today.

## 部署选择

优先比较测试准确率与每类召回率；再结合延迟、显存和训练成本决定。检索方法需计入检索时间；LoRA 合并后延迟可能变化，本实验未测量合并部署。
额外实验用相同模型、相同提示说明、相同4个示例数量比较固定与TF-IDF检索；检索只能访问训练集。提升不能归因于模型大小，因为两个主模型同时在规模和训练方法上存在差异。