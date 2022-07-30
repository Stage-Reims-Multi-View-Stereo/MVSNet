# Generic helpers

import numpy as np
import io

# NOTE: Ajoute automatiquement "\n" à la fin de la chaîne
def array2string_nobrackets(array: np.ndarray) -> str:
    # Source: https://stackoverflow.com/a/42046765
    # Permet de sauvegarder les matrices dans camera_000000[0-15].txt

    bio = io.BytesIO()
    format = "%.18f" # évite la notation scientifique qui n'est peut être pas parsée par MVSNet
    np.savetxt(bio, array, fmt=format)
    saved_string = bio.getvalue().decode("latin1")

    return saved_string

def write_lines(output_file_name: str, lines: list) -> None:

    with open(output_file_name, 'w') as output_file:
        output_file.write("\n".join(lines))

def main():
    array_test = np.arange(10).reshape(5, 2) * 12345.6789
    print(array2string_nobrackets(array_test))

if __name__ == "__main__":
    main()