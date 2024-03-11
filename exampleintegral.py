from scipy import integrate


def f(x):
    # La función irregular
    return abs(x**2 - 1)

maximums = [-1, 0, 1, 2]

def integral(function, maximums):
    results = []
    for i in range(len(maximums) - 1):
        resultado, _ = integrate.quad(function, maximums[i], maximums[i+1])
        results.append(resultado)
    return results

print(integral(f, maximums)) 

