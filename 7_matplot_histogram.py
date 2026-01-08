# ...existing code...
import numpy as np
import matplotlib.pyplot as plt


def main():
    # generate random data (standard normal)
    rng = np.random.default_rng(42)
    data = rng.normal(loc=0.0, scale=1.0, size=1000)

    plt.figure(figsize=(8, 5))
    plt.hist(data, bins=30, color='C0', edgecolor='black', alpha=0.8)
    plt.title('Histogram — Standard Normal (n=1000)')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.6)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()