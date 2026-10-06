import numpy as np


#считаем вектор невязки r = Ax - b и его длину
def compute_r(A, b, x):
    r = A @ x - b
    norm_r = float(np.linalg.norm(r))
    return r, norm_r


#метод гаусса, flag отвечает за выбор главного элемента
def gauss_e(A, b, flag=True):
    n = len(b)
    #делаем расширенную матрицу
    m = np.hstack([A.astype(float), b.astype(float).reshape(-1, 1)])

    #прямой ход
    for k in range(n - 1):
        if flag:
            #ищем максимальный по модулю в столбце
            max_row = k + int(np.argmax(np.abs(m[k:, k])))
            if max_row != k:
                #меняем строки местами
                m[[k, max_row]] = m[[max_row, k]]
        #зануляем элементы под диагональю
        for i in range(k + 1, n):
            factor = m[i, k] / m[k, k]
            m[i, k:] -= factor * m[k, k:]
    #обратный ход
    x = np.zeros(n, dtype=float)
    for i in range(n - 1, -1, -1):
        sum_row = np.dot(m[i, i + 1 : n], x[i + 1 : n])
        x[i] = (m[i, n] - sum_row) / m[i, i]
    return x


#метод простой итерации (якоби)
def simple_i(A, b, x0, eps=1e-4, max_iter=1000, vivod=False):
    n = len(b)
    x_prev = x0.astype(float).copy()
    for it in range(1, max_iter + 1):
        x_next = np.zeros(n, dtype=float)
        for i in range(n):
            #вычитаем диагональный элемент из суммы
            s = np.dot(A[i, :], x_prev) - A[i, i] * x_prev[i]
            x_next[i] = (b[i] - s) / A[i, i]
        #норма разности соседних шагов
        diff = float(np.linalg.norm(x_next - x_prev))
        if vivod:
            print(f"шаг {it}: x = {np.round(x_next, 5)}, delta = {diff:.2e}")
        if diff < eps:
            return x_next, it
        x_prev = x_next.copy()
    return x_prev, max_iter


#метод зейделя
def seidel_m(A, b, x0, eps=1e-4, max_iter=1000, vivod=False):
    n = len(b)
    x = x0.astype(float).copy()

    for it in range(1, max_iter + 1):
        x_old = x.copy()
        for i in range(n):
            #берем уже посчитанные на текущем шаге и старые элементы
            s1 = np.dot(A[i, :i], x[:i])
            s2 = np.dot(A[i, i + 1 :], x_old[i + 1 :])
            x[i] = (b[i] - s1 - s2) / A[i, i]
        diff = float(np.linalg.norm(x - x_old))
        if vivod:
            print(f"шаг {it}: x = {np.round(x, 5)}, delta = {diff:.2e}")
        if diff < eps:
            return x, it
    return x, max_iter


#матрица и правая часть по 4 варианту
A = np.array(
    [
        [13.18, 1.47, -1.25, 2.11],
        [-1.38, 11.56, 2.04, -0.93],
        [1.24, -2.17, 12.37, 1.42],
        [2.16, 0.83, -1.54, 10.68],
    ]
)

b = np.array([21.46, 13.82, 18.95, 16.27])

#определитель и число обусловленности
print(f"det = {np.linalg.det(A):.3f}")
print(f"cond = {np.linalg.cond(A):.3f}")

#считаем через numpy для проверки
x_np = np.linalg.solve(A, b)
print("\nnumpy:", x_np)
print("невязка numpy:", compute_r(A, b, x_np)[1])

#гаусс с выбором ведущего и без
x_g1 = gauss_e(A, b, flag=True)
print("\nгаусс (с выбором):", x_g1)
print("невязка:", compute_r(A, b, x_g1)[1])

x_g2 = gauss_e(A, b, flag=False)
print("гаусс (без выбора):", x_g2)
print("невязка:", compute_r(A, b, x_g2)[1])

#вывод шагов итераций
print("\nход простой итерации (eps=1e-4)")
simple_i(A, b, np.zeros(4), eps=1e-4, vivod=True)

print("\nход зейделя (eps=1e-4)")
seidel_m(A, b, np.zeros(4), eps=1e-4, vivod=True)

#тест с разными начальными точками
print("\nразные x0 (eps=1e-4)")
x0_list = {
    "нули": np.zeros(4),
    "единицы": np.ones(4),
    "произвольный [10, -5, 5, -10]": np.array([10.0, -5.0, 5.0, -10.0]),
}
for name, x0 in x0_list.items():
    _, it_iter = simple_i(A, b, x0, eps=1e-4)
    _, it_seidel = seidel_m(A, b, x0, eps=1e-4)
    print(f"{name} -> итераций: простая = {it_iter}, зейдель = {it_seidel}")

#тест с разной точностью
print("\nразная точность (x0 = нули)")
for eps in [1e-2, 1e-4, 1e-6]:
    x_it, it_iter = simple_i(A, b, np.zeros(4), eps=eps)
    x_sd, it_seidel = seidel_m(A, b, np.zeros(4), eps=eps)
    print(f"eps={eps:.0e} | простая: {it_iter} шагов, невязка {compute_r(A, b, x_it)[1]:.2e}")
    print(f"eps={eps:.0e} | зейдель: {it_seidel} шагов, невязка {compute_r(A, b, x_sd)[1]:.2e}")

#эксперимент с ок углением коэффициентов
print("\nокругление коэффициентов")
#до 1 знака
A_r1 = np.round(A, 1)
b_r1 = np.round(b, 1)
x_r1 = np.linalg.solve(A_r1, b_r1)
print("округление до 1 знака:", x_r1)
print("разница с точным:", np.linalg.norm(x_r1 - x_np))

#до целых
A_r0 = np.round(A, 0)
b_r0 = np.round(b, 0)
x_r0 = np.linalg.solve(A_r0, b_r0)
print("округление до целых:", x_r0)
print("разница с точным:", np.linalg.norm(x_r0 - x_np))