import torch
from torch.distributions import multinomial
from d2l import torch as d2l

# 传入一个概率向量
fair_probs = torch.ones([6]) / 6
# print(multinomial.Multinomial(1, fair_probs).sample())
# print(multinomial.Multinomial(10, fair_probs).sample())

# 模拟1000次投掷
counts = multinomial.Multinomial(1000, fair_probs).sample()
# print(counts/1000)

# 进行500组实验，每组抽取10个样本
counts = multinomial.Multinomial(10, fair_probs).sample((500,))
cum_counts = counts.cumsum(dim=0)
estimates = cum_counts / cum_counts.sum(dim=1, keepdim=True)

d2l.set_figsize((6,4.5))
for i in range(6):
    d2l.plt.plot(estimates[:, i].numpy(), label = ("P(di2=" + str(i + 1) + ")"))
d2l.plt.axhline(y = 0.167, color = 'black', linestyle = 'dashed')
d2l.plt.gca().set_xlabel('Groups of experiments')
d2l.plt.gca().set_ylabel('Estimated probability')
d2l.plt.legend()
# d2l.plt.show()
print(help(d2l.plt.show))