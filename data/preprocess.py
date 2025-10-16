import random
import argparse
import os

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
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Preprocess data file and randomly select k facilities')
    parser.add_argument('filename', type=str, help='Input filename (e.g., small_random.co)')
    parser.add_argument('k', type=int, help='Number of facilities to randomly select')
    parser.add_argument('-o', '--output', type=str, help='Output filename (optional, will be auto-generated if not provided)')

    args = parser.parse_args()

    # Read and process the input file
    data = readfile(args.filename)
    if data:
        min_x = min(pt[1] for pt in data)
        min_y = min(pt[2] for pt in data)
        data = [(vid, x - min_x + 1, y - min_y + 1) for (vid, x, y) in data]

    # Validate k value
    if args.k > len(data):
        print(f"Error: k ({args.k}) cannot be greater than the number of data points ({len(data)})")
        return

    # Randomly select k facilities
    facility_list = random.sample(data, args.k)
    user_list = [item for item in data if item not in facility_list]

    # Generate output filename if not provided
    if args.output:
        output_filename = args.output
    else:
        # Extract base name without extension
        base_name = os.path.splitext(os.path.basename(args.filename))[0].split('.')[1]
        output_filename = f"{base_name}_f{args.k}.co"

    # Write output file
    with open(output_filename, "w") as f1:
        f1.write(f"{len(facility_list)}\n")
        for vid, x, y in facility_list:
            f1.write(f"{vid} {x} {y}\n")
        f1.write(f"{len(user_list)}\n")
        for vid, x, y in user_list:
            f1.write(f"{vid} {x} {y}\n")

    print(f"Successfully processed {args.filename}")
    print(f"Selected {args.k} facilities from {len(data)} total points")
    print(f"Output written to: {output_filename}")


if __name__ == "__main__":
    main()