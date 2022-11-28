import re
from typing import Optional, Iterator

import numpy as np
from scipy import sparse

from ..lib import find_line_in_file, grouped_tmplt, simple_int_tmplt


__all__ = [
    'read_rwfdump',
    'read_ci_vectors',
]


rwfdump_header_patt = re.compile(r'Dump of file\s*%s\s*length\s*%s\s*\(read left to right\):' % (
    grouped_tmplt % (r'code', simple_int_tmplt), grouped_tmplt % (r'size', simple_int_tmplt)
))
ROW_SIZE = 5  # Number of elements in a single row prin


def read_rwfdump(filename: str, n: Optional[int] = None, chunk_size: int = 5,
                 to_numpy: bool = True, dtype=np.float64) -> Iterator[np.ndarray | list[str]]:
    """Read file form RWF dump iteratively, converting on the fly

    Parameters:
        filename: Name of the RWF dump file to read from.
        n: Number of elements to read in total.
        chunk_size: chunk_size * ROW_SIZE elements to yield in one iteration.
        dtype: type to convert elements into.
        to_numpy:

    Returns:
        Generator that yields numpy arrays of type dtype.
    """
    with open(filename, 'r') as file:
        match, line = find_line_in_file(file, rwfdump_header_patt, default_group_map=int)
        if match is None:
            print(line)
            raise ValueError(f"Could not find the header in {filename}")

        code, size = match.pop('code'), match.pop('size')

        n = n if n else size
        if n > size:
            raise ValueError(f"Couldn't read more elements ({n}) than present ({size}) on the file")

        total_lines = -(size // -ROW_SIZE)  # Ceiling division

        lines_read, elems_read = 0, 0
        while lines_read < total_lines and elems_read < n:
            n_lines = min(chunk_size, total_lines - lines_read)

            raw_matrix = []
            for _ in range(n_lines):
                raw_matrix.extend(file.readline().replace('D', 'e').split())

            lines_read += n_lines
            elems_read += len(raw_matrix)

            if to_numpy:
                yield np.asarray(raw_matrix, dtype=dtype)
            else:
                yield raw_matrix


MAX_READ = 1024 ** 2 * 5


def read_ci_vectors(filename: str, n_vecs: int, n_det: int,
                    norm_error: float = 1e-8, max_ndet: int = 50) -> sparse.coo_array:
    """Reads CI vectors from Gaussian's RWF dump of file 635R.

    TODO: Implement reading real CI vectors.
    TODO: Implement reading specific vectors only.

    Parameters:
        filename: Name of the file to read from;
        n_vecs: Total number of CI vectors to read;
        n_det: Number of determinants comprising a single CI vector;
        norm_error: Tolerated error in the norm of CI vector, before vector it is truncated;
        max_ndet: Maximum number of determinants to be stored for CI vector. For now only used to initialize arrays
            for COO constructor;

    Returns:
         Sparse Array in COO format.
    """
    # Arrays to be passed to COO constructor
    data = np.zeros(n_vecs * max_ndet, dtype=np.complex128)
    row_idx = np.zeros(n_vecs * max_ndet, dtype=np.uint64)
    col_idx = np.zeros(n_vecs * max_ndet, dtype=np.uint64)

    norm_cutoff = 1 - norm_error
    n_data = 0  # Number of elements currently stored in data array
    n_vecs_read = 0  # Number of vectors processed and stored in data array

    raw_vec_size = n_det * 2  # Each coefficient consists of real and imaginary part
    buffer = np.zeros(raw_vec_size, dtype=np.float64)  # Array to store raw CI coefficients during reading
    n_buffer = 0  # Number of elements currently stored in buffer

    max_chunk_size = min(raw_vec_size // ROW_SIZE, MAX_READ)
    for chunk in read_rwfdump(filename, chunk_size=max_chunk_size):
        if n_buffer + len(chunk) <= raw_vec_size:
            buffer[n_buffer:n_buffer + len(chunk)] = chunk
            n_buffer += len(chunk)

            if n_buffer < raw_vec_size or n_vecs_read + 1 != n_vecs:
                continue

        # Fill curr_raw_vec to complete currently read CI Vector
        left = raw_vec_size - n_buffer
        buffer[n_buffer:n_buffer + left] = chunk[:left]
        n_buffer += left

        # Construct CI vector and calculate squared norms of its coefficients
        true_vec = buffer[::2] + buffer[1::2] * 1.j
        norm_vec = np.abs(true_vec) ** 2

        # Sort CI coefficients in descending order and calculate cumulative norm
        idx = np.argsort(norm_vec)[::-1]
        cum_norm = np.cumsum(norm_vec[idx])

        # Get CI coefficients to be stored
        stored_idx = idx[cum_norm <= norm_cutoff]
        n_stored = len(stored_idx)

        # If the data array is not big enough for the new vector extend its size
        if n_data + n_stored > len(data):
            data.resize((len(data) * 2,), refcheck=False)
            row_idx.resize((len(row_idx) * 2,), refcheck=False)
            col_idx.resize((len(col_idx) * 2,), refcheck=False)

        # Transfer selected coefficients into data array, keeping track of indexing
        data[n_data:n_data + n_stored] = true_vec[stored_idx]
        row_idx[n_data:n_data + n_stored] = n_vecs_read
        col_idx[n_data:n_data + n_stored] = stored_idx

        # Update counters
        n_vecs_read += 1
        n_data += n_stored

        # Save beginning of the new CI vector from the chunk into buffer
        excess = len(chunk) - left
        buffer[:excess] = chunk[left:]
        n_buffer = excess

    return sparse.coo_array(
        (data[:n_data], (row_idx[:n_data], col_idx[:n_data])),
        shape=(n_vecs, n_det)
    )
