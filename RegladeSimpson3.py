import numpy as np
import matplotlib.pyplot as plt

def simpson_rule(f, a, b, n):
    """Aproxima la integral de f(x) en [a, b] usando la regla de Simpson."""
    if n % 2 == 1:
        raise ValueError("El número de subintervalos (n) debe ser par.")
    
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)  # Puntos del intervalo
    fx = f(x)  # Evaluamos la función en esos puntos
    
    # Regla de Simpson
    integral = (h / 3) * (fx[0] + 2 * np.sum(fx[2:n:2]) + 4 * np.sum(fx[1:n:2]) + fx[n])
    
    return integral

# Parámetros del problema
k = 0.5  # W/m-K
x1 = 0  # m
x2 = 2  # m

# Función de temperatura
def temperatura(x):
    return 300 - 50 * x**2

# Derivada de la temperatura (dT/dx)
def derivada_temperatura(x):
    return -100 * x

# Función a integrar (flujo de calor)
def flujo_calor(x):
    return derivada_temperatura(x)

# Solución analítica (para comparar)
solucion_analitica = k * (temperatura(x2) - temperatura(x1))

# Valores de n para la regla de Simpson
n_valores = [6, 10, 20, 30]

# Resultados numéricos y errores
resultados = []
errores = []

for n in n_valores:
    resultado_num = k * simpson_rule(flujo_calor, x1, x2, n)
    resultados.append(resultado_num)
    error = abs(resultado_num - solucion_analitica)
    errores.append(error)
    print(f"n = {n}: Flujo de calor = {resultado_num:.4f} W/m², Error = {error:.4f} W/m²")

print(f"\nSolución analítica: {solucion_analitica:.4f} W/m²")

# Gráfica de la función y la aproximación (con n = 30)
x_vals = np.linspace(x1, x2, 100)
Q_vals = flujo_calor(x_vals)

plt.plot(x_vals, Q_vals, label=r"$Q(x) = \frac{dT}{dx}$", color="blue")
plt.fill_between(x_vals, Q_vals, alpha=0.3, color="cyan", label="Área aproximada")
plt.scatter(np.linspace(x1, x2, 30 + 1), flujo_calor(np.linspace(x1, x2, 30 + 1)), color="red", label="Puntos de interpolación")
plt.xlabel("Posición (m)")
plt.ylabel("Flujo de calor (W/m²)")
plt.legend()
plt.title("Flujo de calor a través de la pared (Regla de Simpson)")
plt.grid()

# Guardar la figura
plt.savefig("flujo_calor_simpson.png")
plt.show()

# Gráfica del error en función de n
plt.plot(n_valores, errores, marker='o')
plt.xlabel("Número de subintervalos (n)")
plt.ylabel("Error (W/m²)")
plt.title("Error en la aproximación del flujo de calor")
plt.grid()

# Guardar la figura
plt.savefig("error_flujo_calor.png")
plt.show()
