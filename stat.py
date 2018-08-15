

#################################
Safe and nested subgame solving…
#################################
非完美信息和完美信息的区别？
	子博弈之间会相互影响，并非独立。

Unsafe subgame是什么？
	进入S_top的概率是所有单步转移概率的算术权，不合理

subgame solving？
	引入反事实最优反应CBR，还有反事实最优反应价值CBV。
	solving方法引入CBV边际，通过最大化最小的CBV边际来求解。

reach subgame solving？
	把子博弈独立开来考虑，边际最大化会导致次优。
	希望考虑到对手在子博弈中所得到的。
	考虑到子博弈的策略会对自己造成的影响
	1.引入不同到达路径的CBV增长价值——gift
	2.结合gift的定义，给出一个信息集下的gitf的lower bound，加入前面的CBV边际求解方法，得到了CBV边际的增长非负性质。并把这个叫做Reach-Maxmargin
	3.证明在独立解决subgame的情况下，reach-maxmargin是一个不会比Maxmargin差的选择

Nested subgame solving？
	拓展到更高维或者连续空间情况下的解决方法
	safe的条件被放弃了，加入了一个新的CBV的值来做空间的映射。
	作者说明了 这个Inexpensive的方法不能用在unsafe subgame solving方法得到的子博弈结果——也就是blue print还有maxmargin
	（也就是说，虽然单独的子博弈的reach-Max..是safe的，但是实际上拓展到高维空间的时候并非safe。）

实证
	证明了是在inexpensive拓展+reach-Maxmargin是最好的


#################################
Financial Trading as a Game:
A Deep Reinforcement Learning Approach
#################################

DRQN = DQN + RNN

修改了MDP使其适合金融数据，怎么做
	分成三个数据部分
	1.时间，转换为sin
	2.市场特征
	3.仓位特征
修改Q-learnning算法
	看不懂到底改了什么emmmm????
找到适合DRQN的超参数
	不同于传统其他的算法，用小的replay memory更有效。That is，金融数据更多是短视的
	RNN上面，用更长的采样长度。并由于采样长度的原因可以不需要每步都进行训练。
	丢掉了Forwalk-walk opt process，因为会过拟合
行为提升算法来减少搜索空间
	就是直接不需要你去搜索了，直接列举出了所有的状态空间的收益

几个研究方向
	1.拓宽特征空间还有决策空间
	2.用在多个策略的决策上
	3.用分布式RL，就不需要计算E（这纯粹是为了简化计算），就可以直接算出Q


#################################
Market Self-Learning of Signals, Impact and Optimal Trading:
Invisible Hand Inference with Free Energy
(or, How We Learned to Stop Worrying and Love Bounded Rationality)
#################################
文章结果
	用RL模拟整个不完美的市场，并且发现均值回归现象是合理的

假设
	1.所有人服从Markowitz E-V理论来管理组合


这篇的其他细节比较多，引用的文献也很多，需要很长时间才能啃下来。
只关注一个问题，RL到底在这里面起到了什么作用？

#################################
Robust Log-Optimal Strategy with Reinforcement Learning
#################################

GLOS 广义的对数优化策略
	优点是有比较好的数学性质 

RL的用处
	用来结合之前有的各种组合优化方法
	但根本用的不是RL，只用了CNN的方法来做每一步的权值预测

#################################
Risk-Aware Multi-Armed Bandit Problem with
Application to Portfolio Selection
#################################



#################################
Agent Inspired Trading Using Recurrent
Reinforcement Learning and LSTM Neural Networks
#################################



#################################
Solving High-Dimensional Partial Differential Equations Using deep learning
#################################


#################################
Modelling Stock-market Investors as
Reinforcement Learning Agents
#################################


#################################
Reinforcement learning in market games
#################################

Q-Learning and SARSA:a comparison between two intelligent stochastic 
control approaches for financial trading

MACHINE LEARNING FOR TRADING

Factor Selection with Deep Reinforcement Learning for Financial Forecasting

Financial Planning via Deep Reinforcement Learning AI


