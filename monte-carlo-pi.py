import random
import matplotlib.pyplot as plt
import numpy as np

n = 100_000
inside_circle = 0
points_inside_circle = []
points_outside_circle = []
line_graph = []

for i in range(1, n + 1):
    x = random.uniform(-1, 1)
    y = random.uniform(-1, 1)
    if x**2 + y**2 <= 1:
        points_inside_circle.append((x, y))
        inside_circle += 1
    else:
        points_outside_circle.append((x, y))
    
    pi_approx = inside_circle / i * 4
    line_graph.append(-np.log10(abs(pi_approx - np.pi)))

# Plot points
plt.scatter([p[0] for p in points_inside_circle], [p[1] for p in points_inside_circle], color='green', s=1)
plt.scatter([p[0] for p in points_outside_circle], [p[1] for p in points_outside_circle], color='red', s=1)
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.title(f'Monte Carlo Pi: {inside_circle / n * 4}')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()

plt.semilogx(range(1, n + 1), line_graph)
plt.show()

print(inside_circle / n * 4)
