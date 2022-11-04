import numpy as np
import matplotlib.pyplot as plt

class read_data:
    def __init__(self):
        self.columns = ''

    def data_frame(self, file):
        with open(file,'r') as data:
            a = [text.split('\n')[0] for text in data.readlines()]
            self.columns = a[0].split(',')
            return {feature:np.array([float(text.split(',')[idx]) for text in a[1:]]) for idx, feature in enumerate(self.columns)}
    
    def show(self):
        [print(f"|{i}|") for i in self.columns]

class LR:
    def __init__(self):
        self.w = np.array(10)
        self.b = np.array(10)

    def hypothesis(self, x):
        return np.dot(self.w,x) + self.b

    def cost(self, x, y):
        return (1/(2*len(x)))*(np.sum(np.power(self.hypothesis(x) - y,2)))

    def fit(self, x, y, iterations, alpha = 0.1):
        a = []
        for _ in range(iterations):
            J = self.cost(x,y)
            self.w = self.w - (alpha/len(x))*np.sum((self.hypothesis(x)-y)*x)
            self.b = self.b - (alpha/len(x))*np.sum(self.hypothesis(x) - y)
            a.append(J)
        return self.w, self.b

    def plot(self,x,y):
        t = np.arange(np.min(x), np.max(x))
        plt.scatter(x,y)
        plt.plot(t,self.hypothesis(t))
        plt.show()