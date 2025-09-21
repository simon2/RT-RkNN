import random

def generate_coordinates(num_points, filename="small_random.co"):
    with open(filename, 'w') as f:
        f.write(f"p {num_points}\n")
        for i in range(1, num_points + 1):
            x = random.randint(0, 1000)
            y = random.randint(0, 1000)
            f.write(f"v {i} {x} {y}\n")

generate_coordinates(100)