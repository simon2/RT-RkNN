import random

# Generate data
with open('test10.txt', 'w') as f:
    f.write("100\n")  # Number of data points
    for i in range(1, 11):
        x = random.randint(0, 100)
        y = random.randint(0, 100)
        f.write(f"{i} {x} {y}\n")

print("Data written to test10.txt")