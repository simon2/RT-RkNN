import random

def readfile(filename):
    result = []
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith("c"):
                continue

            elif line.startswith("p"):
                parts = line.split()
                n = int(parts[-1])
                result = [None] * n
                idx = 0

            elif line.startswith("v"):
                parts = line.split()
                vid = int(parts[1])
                x = int(parts[2])
                y = int(parts[3])

                if idx < len(result):
                    result[idx] = (vid, x, y)  # 存为元组
                    idx += 1
    return result

def main():
    data = readfile("small_random.co")
    if data:
        min_x = min(pt[1] for pt in data)
        min_y = min(pt[2] for pt in data)
        data = [(vid, x - min_x, y - min_y) for (vid, x, y) in data]

    k = 20  # predetermined length
    facility_list = random.sample(data, k)  # randomly choose k items
    user_list = [item for item in data if item not in facility_list]
    
    with open("small_random_20.co", "w") as f1:
        f1.write(f"{len(facility_list)}\n")
        for vid, x, y in facility_list:
            f1.write(f"{vid} {x} {y}\n")
        f1.write(f"{len(user_list)}\n")
        for vid, x, y in user_list:
            f1.write(f"{vid} {x} {y}\n")


if __name__ == "__main__":
    main()