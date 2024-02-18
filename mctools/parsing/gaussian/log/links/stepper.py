from __future__ import annotations

import numpy as np

from parsing.core import LineStepper


class LogStepper(LineStepper):
    N_COLUMNS_PER_BLOCK: int = 5

    def read_square_matrix(self, dtype=np.float_) -> np.ndarray:
        array = []
        dim = 0
        self.step_forward()
        while '.' in (line := self.readline()):
            row = self.line_to_data(line)
            array.extend(map(float, row[1:]))
            dim += 1

        if len(array) == dim * self.N_COLUMNS_PER_BLOCK:
            array = self.read_rectangular(dim, dim, array=array)
            matrix = np.asarray(array, dtype=dtype).reshape(dim, dim)
        else:
            matrix = np.asarray(self.read_hermitian(dim, array=array), dtype=dtype)
            # matrix = np.zeros((dim, dim), dtype=dtype)
            # tril = np.tril_indices(dim)
            # matrix[tril] = array
            # matrix += matrix.T - matrix.diagonal()
        self.step_back()
        return matrix

    def read_hermitian(self, dim: int, array: list[float]) -> list[float]:
        array = [] if array is None else array
        for col_jdx in range(self.N_COLUMNS_PER_BLOCK, dim, self.N_COLUMNS_PER_BLOCK):
            for row_jdx in range(col_jdx, dim):
                row = self.line_to_data(self.readline())
                array.extend(map(float, row[1:]))
                #  First value of the row   ^ is in a row index
            self.step_forward()
        return array

    def read_rectangular(self, n_rows: int, n_cols: int, /, array: list[float] = None) -> list[float]:
        array = [] if array is None else array
        for col_jdx in range(0, n_cols, self.N_COLUMNS_PER_BLOCK):
            self.step_forward()  # Skip header row
            for row_jdx in range(n_rows):
                row = self.line_to_data(self.readline())
                array.extend(map(float, row[1:]))
                #  First value of the row   ^ is in a row index
        return array

    @staticmethod
    def line_to_data(line) -> list[float]:
        return line.strip().replace('D', 'e').split()
