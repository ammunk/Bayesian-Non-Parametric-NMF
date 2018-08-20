import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from trandn import trandn

n_samples = int(1e6)
mu = -2000
std = 0.3
mus = np.zeros(n_samples) + mu
stds = np.zeros(n_samples) + std
lower_bound = np.zeros(n_samples)
upper_bound = np.zeros(n_samples) + np.infty
samples = trandn((lower_bound-mus)/stds, (upper_bound-mus)/stds)
samples = samples*stds + mus
mean = samples.mean()

alpha = -mu/std
alpha_pdf = norm.pdf(alpha)
Z = 1 - norm.cdf(alpha)
theoretical_mean = mu + std*(alpha_pdf/Z)


print(f"Estimated mean: {mean}\nTheoretical mean: {theoretical_mean}")

fig = plt.figure(figsize=(20,20))
sns.distplot(samples, kde=False)
plt.show()
