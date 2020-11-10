import pandas as pd
import matplotlib.pyplot as plt

df = pd.DataFrame({"Setup": ['Baseline', 'BPE', 'BPE_dropout', 'BPE_dropout_iss'],
                   "BLEU": [5.9, 19.4, 18.4, 10.6],
                   "unigram probability": [37.5, 54.0, 53.9, 46.6],
                   "bigram probability": [8.3, 25.6, 25.4, 15.9],
                   "trigram probability": [3.1, 14.7, 14.7, 6.9]
                   })

fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)
df.plot(y=["BLEU"], x="Setup", kind="bar", ax=axes[0,0], colormap=plt.get_cmap('spring'))
df.plot(y=["unigram probability"], x="Setup", kind="bar", ax=axes[0,1], colormap=plt.get_cmap('summer'))
df.plot(y=["bigram probability"], x="Setup", kind="bar", ax=axes[1,0], colormap=plt.get_cmap('autumn'))
df.plot(y=["trigram probability"], x="Setup", kind="bar", ax=axes[1,1], colormap=plt.get_cmap('winter'))

plt.show()
