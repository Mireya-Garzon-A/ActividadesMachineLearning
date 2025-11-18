import numpy as np
import pickle
from entorno import GridWorld

class QLearning:
    def __init__(self, env, alpha=0.1, gamma=0.95, epsilon=1.0):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = np.zeros((env.size**2, 4))  # 25 estados, 4 acciones

    def entrenar(self, episodios=300):
        recompensas = []
        for _ in range(episodios):
            s = self.env.reset()
            total = 0
            for _ in range(100):
                a = np.random.randint(4) if np.random.rand() < self.epsilon else np.argmax(self.Q[s])
                s2, r, done = self.env.step(a)
                self.Q[s,a] += self.alpha * (r + self.gamma * np.max(self.Q[s2]) - self.Q[s,a])
                s = s2
                total += r
                if done: break
            recompensas.append(total)
            self.epsilon *= 0.99  # decaimiento
        return recompensas

    def trayectoria(self):
        s = self.env.reset()
        camino = [s]
        for _ in range(100):
            a = np.argmax(self.Q[s])
            s, _, done = self.env.step(a)
            camino.append(s)
            if done: break
        return camino

    def guardar(self, ruta="modelo.pkl"):
        with open(ruta, "wb") as f: pickle.dump(self.Q, f)

    def cargar(self, ruta="modelo.pkl"):
        with open(ruta, "rb") as f: self.Q = pickle.load(f)
