import pickle
import matplotlib.pyplot as plt

train_loss = pickle.load(file=open('train_loss.pickle', 'rb'))
val_loss = pickle.load(file=open('val_loss.pickle', 'rb'))
plt.ylim((0, 10))
plt.plot(range(len(val_loss)), val_loss, range(len(train_loss)), train_loss)
plt.show()