import matplotlib.pyplot as plt
import numpy as np
x=np.linspace(0,2*(np.pi),100)
y_clean=np.sin(x)
plt.plot(x,y_clean,label='clean signal',color='m')
y_noise=np.random.normal(0,0.2,100)
plt.plot(x,y_noise,label='noise',color='c')
y_noisy=y_clean+y_noise
plt.plot(x,y_noisy,label='noisy signal',color='k')
plt.title('plot of clean and noisy signal')
plt.xlabel('time')
plt.ylabel('amplitude')
plt.savefig('signal_plot_1.png')
plt.show()