import matplotlib.pyplot as plt
import numpy as np

def read_coordinates(filename):
    """Read coordinates from DIMACS format file"""
    coords = []
    
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.strip().split()
                # parts[0] is 'v', parts[1] is id, parts[2] is x, parts[3] is y
                node_id = int(parts[1])
                x = int(parts[2])
                y = int(parts[3])
                coords.append((node_id, x, y))
    
    return coords

def plot_points(coords, output_filename='map_points.png', point_size=0.1):
    """Plot points and save as PNG"""
    
    # Extract x and y coordinates
    x_coords = [c[1] for c in coords]
    y_coords = [c[2] for c in coords]
    
    # Create figure with white background
    fig, ax = plt.subplots(figsize=(12, 10), dpi=100)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # Plot points as black dots
    ax.scatter(x_coords, y_coords, s=point_size, c='black', marker='.')
    
    # Remove axes for cleaner look
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Set equal aspect ratio to preserve shape
    ax.set_aspect('equal', adjustable='box')
    
    # Tight layout to minimize white space
    plt.tight_layout()
    
    # Save as PNG
    plt.savefig(output_filename, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"Image saved as {output_filename}")
    print(f"Total points plotted: {len(coords)}")

def main():
    # Input and output filenames
    input_file = 'USA-road-d.USA.co'  # Replace with your actual filename
    output_file = 'USA.png'
    
    # Read coordinates
    coords = read_coordinates(input_file)
    
    if coords:
        # Plot and save
        plot_points(coords, output_file, point_size=0.1)
        
        # Print some statistics
        x_values = [c[1] for c in coords]
        y_values = [c[2] for c in coords]
        print(f"X range: [{min(x_values)}, {max(x_values)}]")
        print(f"Y range: [{min(y_values)}, {max(y_values)}]")
    else:
        print("No coordinates found in the file")

if __name__ == "__main__":
    main()