import numpy as np

def random_coord(n, x, y, r):
    coord = np.random.randn(n, 2)
    coord *= r
    coord[:, 0] += x
    coord[:, 1] += y
    return coord


base_node_str = "\\draw node[{}, inner sep=0pt, minimum size=5pt, fill, color={}] at ({}, {}) {{}};"
# print(base_node_str)
base_cross_str = "\draw ({x},{y}) node[cross=3pt] {{}};\n\draw ({x},{y}) node [below left]{{$\\rmU_{i}$}};"

if __name__ == "__main__":
    base_coordinates = [(1, 3.5), (4, 4), (3.5, 1)]

    radius = 0.4
    n_elm = 10
    shapes = ["star, star points=5", "diamond", "circle"]
    colors = ["red", "blue", "green"]

    for i in range(len(base_coordinates)):
        coord = base_coordinates[i]
        shape = shapes[i]
        color = colors[i]
        cordinates = random_coord(n_elm, *coord, radius)
        for co in cordinates:
            str = base_node_str.format(shape, color, co[0], co[1])
            print(str)
        print(base_cross_str.format(x=coord[0]+radius/2, y=coord[1]+radius/2, i=i))
