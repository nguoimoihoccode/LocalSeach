import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2

class Problem:
    def __init__(self, filename):
        self.X, self.Y, self.Z = self.load_state_space(filename)
        self.X, self.Y = np.meshgrid(self.X, self.Y)

    def load_state_space(self, filename):
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
        img = cv2.GaussianBlur(img, (5, 5), 0)
        h, w = img.shape
        X = np.arange(w)
        Y = np.arange(h)
        Z = img
        return X, Y, Z

    def show(self):
        fig = plt.figure(figsize=(8,6))
        ax = plt.axes(projection='3d')
        ax.plot_surface(self.X, self.Y, self.Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
        plt.show()

    def draw_path(self, path):
        fig = plt.figure(figsize=(8,6))
        ax = plt.axes(projection='3d')
        ax.plot_surface(self.X, self.Y, self.Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
        path_x, path_y, path_z = zip(*path)
        ax.plot(path_x, path_y, path_z, 'r-', zorder=5, linewidth=2)
        plt.show()
        
    def get_neighbors(self, x, y):
        directions = [(0, -1), (1, 0), (0, 1), (-1, 0), (-1, 1), (1, 1), (-1, -1), (1, -1)]
        neighbors = []
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            # Ensure we are still within the bounds of the state space
            if 0 <= nx < self.Z.shape[1] and 0 <= ny < self.Z.shape[0]:
                neighbors.append((nx, ny, self.Z[ny, nx]))
        return neighbors


