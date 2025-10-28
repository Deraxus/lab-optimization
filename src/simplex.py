import numpy as np

import numpy as np

class SimplexResult:
    def __init__(self, status, obj_value=None, solution=None, basis=None, message=""):
        self.status = status          # "optimal", "infeasible", "unbounded", "failed"
        self.obj_value = obj_value    # значение целевой функции
        self.solution = solution      # вектор переменных (по всем столбцам)
        self.basis = basis            # индексы базисных переменных
        self.message = message


class SimplexTableau:
    def __init__(self, A, b, c, basis):
        """
        A: матрица ограничений (m x n)
        b: свободные члены (m,)
        c: коэффициенты целевой функции (длина n). Мы решаем max c^T x.
        basis: список индексов базисных переменных длины m (каждый индекс от 0..n-1)
        """
        self.m, self.n = A.shape
        self.basis = basis[:]  # текущие базисные переменные

        # Таблица [A | b]
        self.tableau = np.hstack([A.astype(float), b.reshape(-1, 1).astype(float)])

        # Целевая
        self.c = np.array(c, dtype=float)

        # Построим строку цели z_row сразу
        self._rebuild_objective()

    def _rebuild_objective(self):
        """
        Перестраивает строку цели (reduced costs) в виде:
        z_j = c_j - c_B^T * B^{-1} * A_col_j
        z_const = c_B^T * B^{-1} * b
        Это стандарт для максимизации.
        После этого положительные коэффициенты в z_row означают,
        что мы можем улучшить целевую функцию, введя эту переменную в базис.
        """
        # Матрица базисных столбцов B и небазисных тоже нам нужна
        A = self.tableau[:, :self.n]       # m x n
        b = self.tableau[:, self.n]        # m

        B = A[:, self.basis]               # m x m
        B_inv = np.linalg.inv(B)

        cB = self.c[self.basis]            # коэффициенты цели для базисных
        # Значение цели при текущем базисе
        obj_val = cB @ (B_inv @ b)

        # reduced costs для всех переменных
        z_row = np.zeros(self.n + 1)
        for j in range(self.n):
            col_j = A[:, j]                # столбец A_j
            z_row[j] = self.c[j] - cB @ (B_inv @ col_j)

        # свободный член (значение целевой ф-ции)
        z_row[self.n] = obj_val

        self.z_row = z_row

    def _choose_entering_variable(self):
        """
        В симплексе для max:
        Выбираем переменную с положительным reduced cost (максимально положительным).
        Если все <= 0 → оптимум.
        """
        costs = self.z_row[:self.n]
        j = np.argmax(costs)
        if costs[j] <= 1e-9:
            return None  # улучшать нельзя, оптимум
        return j

    def _choose_leaving_variable(self, enter_j):
        """
        Правило минимального отношения b_i / a_i_j для положительных a_i_j.
        Если все a_i_j <= 0 → целевая неограниченно растёт вдоль этого направления.
        """
        col = self.tableau[:, enter_j]
        b_vals = self.tableau[:, self.n]
        ratios = []
        for i in range(self.m):
            if col[i] > 1e-12:
                ratios.append(b_vals[i] / col[i])
            else:
                ratios.append(np.inf)

        i_min = int(np.argmin(ratios))
        if ratios[i_min] == np.inf:
            return None  # неограниченно
        return i_min

    def _pivot(self, pivot_i, pivot_j):
        """
        Поворот вокруг элемента tableau[pivot_i, pivot_j].
        Делает переменную pivot_j базисной в строке pivot_i.
        """
        # Нормируем опорную строку
        pivot_val = self.tableau[pivot_i, pivot_j]
        self.tableau[pivot_i, :] /= pivot_val

        # Зануляем этот столбец в остальных строках
        for i in range(self.m):
            if i == pivot_i:
                continue
            factor = self.tableau[i, pivot_j]
            self.tableau[i, :] -= factor * self.tableau[pivot_i, :]

        # Обновляем базис
        self.basis[pivot_i] = pivot_j

        # Пересобираем строку цели
        self._rebuild_objective()

    def print_table(self, title=""):
        print("\n" + title)
        print("Базис | ", end="")
        for j in range(self.n):
            print(f"x{j+1:>3} ", end="")
        print(" |  b ")
        print("-" * (7 + 6*self.n))

        for i in range(self.m):
            basis_var = f"x{self.basis[i]+1}"
            row = " ".join(f"{v:6.2f}" for v in self.tableau[i])
            print(f"{basis_var:>6} | {row}")
        print("-" * (7 + 6*self.n))
        print("   Z  | " + " ".join(f"{z:6.2f}" for z in self.z_row))


    def solve(self, max_iter=100):
        """
        Делает простой симплекс.
        Возвращает SimplexResult.
        """
        for _ in range(max_iter):
            enter_j = self._choose_entering_variable()

            # Если None — оптимум
            if enter_j is None:
                # Значение целевой функции:
                obj = self.z_row[self.n]

                # Восстанавливаем полный вектор решения
                solution = np.zeros(self.n)
                for row_i, var_j in enumerate(self.basis):
                    solution[var_j] = self.tableau[row_i, self.n]

                return SimplexResult(
                    status="optimal",
                    obj_value=obj,
                    solution=solution,
                    basis=self.basis[:],
                    message="Optimal solution found."
                )

            leave_i = self._choose_leaving_variable(enter_j)
            if leave_i is None:
                # неограниченная цель
                return SimplexResult(
                    status="unbounded",
                    message="Unbounded objective."
                )

            self._pivot(leave_i, enter_j)

        self.print_table("Финальная таблица симплекса")

        return SimplexResult(
            status="failed",
            message="Max iterations exceeded."
        )


def phase_one():
    """
    Фаза I для задачи №18.

    Переменные: [x1,x2,x3,x4,s1,s3,a2,a3]
    Ограничения:
    1) x1 + 2x2 + x3      + s1                   = 11
    2) x1      + x3 + x4            + a2         = 8
    3)      x2      + x4    - s3          + a3   = 3

    Базис стартовый: s1, a2, a3 → индексы [4,6,7]
    Цель фазы I: минимизировать a2 + a3
    => эквивалентная цель для max: W = -(a2 + a3)
    => коэффициенты c для max: [0,0,0,0,0,0,-1,-1]
    """

    A = np.array([
        [1, 2, 1, 0, 1,  0, 0, 0],
        [1, 0, 1, 1, 0,  0, 1, 0],
        [0, 1, 0, 1, 0, -1, 0, 1]
    ], dtype=float)

    b = np.array([11, 8, 3], dtype=float)

    basis = [4, 6, 7]  # s1, a2, a3

    c_phase1 = np.array([0, 0, 0, 0, 0, 0, -1, -1], dtype=float)

    tab = SimplexTableau(A, b, c_phase1, basis)
    res1 = tab.solve()
    return res1, tab



def phase_two(phase1_result, tab_phase1):
    """
    Фаза II: решаем исходную задачу минимизации Z = 2x1 + x2 + 3x3 + 2x4
    (т.е. максимизируем -Z).
    """

    if phase1_result.status != "optimal":
        return SimplexResult(status="infeasible", message="Phase I failed.")

    # Проверяем, что сумма искусственных = 0
    if abs(phase1_result.obj_value) > 1e-7:
        return SimplexResult(status="infeasible", message="Artificial vars not eliminated.")

    # Берём таблицу после фазы I
    A_full = tab_phase1.tableau[:, :tab_phase1.n]
    b_full = tab_phase1.tableau[:, tab_phase1.n]

    # Убираем столбцы искусственных переменных (a2,a3) -> индексы 6,7
    keep_cols = [0, 1, 2, 3, 4, 5]  # x1,x2,x3,x4,s1,s3
    A2 = A_full[:, keep_cols]
    b2 = b_full.copy()

    # Новый базис — выкинем искусственные, заменим их на ближайшие подходящие
    basis2 = []
    for j in tab_phase1.basis:
        if j in [6, 7]:
            # если искусственная, подставим слэк или любую небазисную переменную
            repl = 4 if 4 not in basis2 else 5
            basis2.append(repl)
        else:
            basis2.append(keep_cols.index(j))

    # Целевая функция для max(-Z)
    c_phase2 = np.array([-2, -1, -3, -2, 0, 0], dtype=float)

    tab2 = SimplexTableau(A2, b2, c_phase2, basis2)
    res2 = tab2.solve()
    return res2

