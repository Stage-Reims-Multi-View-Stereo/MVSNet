def read_lines(filename):
    with open(filename) as f:
        return f.read().splitlines()