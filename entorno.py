import numpy as np

ACTIONS = {
    0: (0, 1),   # derecha
    1: (0, -1),  # izquierda
    2: (1, 0),   # abajo
    3: (-1, 0)   # arriba
}

class GridWorld:
    def __init__(self, size=5, start=(0, 0), goal=(4, 4), holes=[(1, 3), (3, 1)]):
        self.size = size
        self.start = start
        self.goal = goal
        self.holes = holes
        self.state = start

    # Reinicia episodio → devuelve ID lineal
    def reset(self):
        self.state = self.start
        return self._sid(self.state)

    # Avanza un paso
    def step(self, action):
        dr, dc = ACTIONS[action]
        r, c = self.state[0] + dr, self.state[1] + dc

        # Limitar a los bordes del grid
        r = max(0, min(self.size - 1, r))
        c = max(0, min(self.size - 1, c))

        self.state = (r, c)
        sid = self._sid(self.state)

        # Condiciones terminales
        if self.state in self.holes:
            return sid, -1.0, True
        if self.state == self.goal:
            return sid, 1.0, True

        # Movimiento normal
        return sid, -0.01, False

    # Codifica estado → entero
    def _sid(self, s):
        return s[0] * self.size + s[1]

    # Decodifica entero → (fila, columna)
    def decode(self, sid):
        return (sid // self.size, sid % self.size)
