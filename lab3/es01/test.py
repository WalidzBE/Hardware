import matplotlib.pyplot as plt

def plot_example():
    plt.figure()
    plt.plot([0, 1, 2, 3], [0, 1, 0, 1], label='Example curve')
    plt.title("Simple Matplotlib Example")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()  # Opens a GUI window

plot_example()
