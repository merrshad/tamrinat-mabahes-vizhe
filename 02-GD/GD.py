
def f(x):
    return (x - 1)**2 + 2


def df(x):
    return 2 * (x - 1)

x = 3  
alpha = 0.1  
tolerance = 1e-6  
max_iter = 1000  


for i in range(max_iter):
    grad = df(x)
    new_x = x - alpha * grad
    if abs(new_x - x) < tolerance:
        break
    x = new_x

print(f"Minimum at x = {x}")
print(f"Minimum value f(x) = {f(x)}")
print(f"Number of iterations: {i+1}")
