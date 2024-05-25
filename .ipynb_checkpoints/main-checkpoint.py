import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

matplotlib.use('Qt5Agg')

# Создание двуз кругов
circle, circle1 = [], []
for _ in range(10000):
    t = np.random.uniform(0, 2 * np.pi, 2)
    X1 = np.cos(t[0])
    y1 = np.sin(t[0])
    circle.append([X1, y1])

    X2 = np.cos(t[1]) + 1
    y2 = np.sin(t[1])
    circle1.append([X2, y2])

# Объединение двух рядов, присвоение класса
X = circle + circle1
y = [1] * 10000 + [2] * 10000

# Разделение ряда на тренировочную и тестовую выборку
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# Градиентный бустинг
model = GradientBoostingClassifier()
model.fit(X_train, y_train)

# Предсказание на тестовой выборке
y_pred = model.predict(X_test)

# Оценка точности
accuracy = accuracy_score(y_test, y_pred, normalize=False)
print("Accuracy:", accuracy)

# Визуализируем точки
# Зеленый - первый ряд, Серый - второй ряд
# Красный - Предсказанные точки первого ряда, Синий - предсказанные точки второго ряда
plt.figure(figsize=(6, 6))
plt.scatter(*zip(*circle), s=50, c='green')
plt.scatter(*zip(*circle1), s=50, c='gray')

# Построение предсказанных точек
for i, x in enumerate(X_test):
    clr = (1, 0, 0) if y_pred[i] == 1 else (0, 0, 1)
    plt.scatter(x[0], x[1], c=clr, s=1)

plt.axis('equal')
plt.show()
