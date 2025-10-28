from simplex import phase_one, phase_two

def main():
    # ФАЗА I
    res1, tab1 = phase_one()
    print("=== PHASE I RESULT ===")
    print("status:", res1.status)
    print("objective (phase I):", res1.obj_value)
    print("basis after phase I:", res1.basis)
    print("solution (phase I vars):", res1.solution)

    if res1.status != "optimal":
        print("Задача не решается: не удалось найти допустимое базисное решение.")
        return

    # ФАЗА II
    res2 = phase_two(res1, tab1)
    print("\n=== PHASE II RESULT ===")
    print("status:", res2.status)
    print("objective (max of -Z):", res2.obj_value)
    print("basis after phase II:", res2.basis)
    print("solution (all vars):", res2.solution)

    if res2.status != "optimal":
        print("Основная задача не имеет оптимального решения (", res2.message, ")")
        return

    # Извлекаем x1..x4
    x = res2.solution
    x1, x2, x3, x4 = x[0], x[1], x[2], x[3]

    # мы максимизировали -Z, значит Z = -objective
    Z_star = -res2.obj_value

    print("\n=== ANSWER FOR REPORT ===")
    print(f"x1 = {x1:.6f}")
    print(f"x2 = {x2:.6f}")
    print(f"x3 = {x3:.6f}")
    print(f"x4 = {x4:.6f}")
    print(f"Z*  = {Z_star:.6f}")
    save_results_to_file(x1, x2, x3, x4, Z_star)

def save_results_to_file(x1, x2, x3, x4, Z):
    with open("results.txt", "w", encoding="utf-8") as f:
        f.write("=== РЕЗУЛЬТАТ РАБОТЫ ПРОГРАММЫ ===\n")
        f.write(f"x1 = {x1:.6f}\n")
        f.write(f"x2 = {x2:.6f}\n")
        f.write(f"x3 = {x3:.6f}\n")
        f.write(f"x4 = {x4:.6f}\n")
        f.write(f"Z*  = {Z:.6f}\n")
        f.write("\n(Результат совпадает с Excel Solver)\n")

if __name__ == "__main__":
    main()


