import argparse
import mmap
import pathlib
import re

import pprint

from typing import BinaryIO, NoReturn, Sequence
from collections import namedtuple

import numpy as np
import numpy.typing as npt

from tqdm import tqdm

from parsing import search_in_file

from parsing.gaussian.log.utils import parse_gdvlog

from parsing.gaussian.log.links.l910 import (
    rdm_real_patt, rdm_imag_patt,
    read_mc_spec
)

rdm_start_patt = re.compile(rb'^ *Final 1PDM for State:\s+(?P<state>\d+) *$', re.MULTILINE)
rdm_real_patt_b = re.compile(rdm_real_patt.pattern.encode())
rdm_imag_patt_b = re.compile(rdm_imag_patt.pattern.encode())


def read_rect_matrix_fast(file: BinaryIO | mmap.mmap, matrix: npt.NDArray) -> NoReturn:
    n_rows, n_cols = matrix.shape

    # Read Gaussian-printed matrix by column chunks
    curr_col = 0
    while curr_col < n_cols:
        line = file.readline()
        next_col = curr_col + len(line.strip().split())
        for i in range(n_rows):
            i, *values = file.readline().replace(b'D', b'E').strip().split()
            matrix[int(i) - 1, curr_col:next_col] = list(map(float, values))
        curr_col = next_col


def read_rdm_fast(file: BinaryIO | mmap.mmap, rdm: npt.NDArray):
    # Read Real
    search_in_file(file, rdm_real_patt_b, n_skips=0, err_msg=f"Couldn't find matrix header: {rdm_real_patt_b.pattern}")
    read_rect_matrix_fast(file, rdm[..., 0])

    # Read Imaginary
    search_in_file(file, rdm_imag_patt_b, err_msg=f"Couldn't find matrix header: {rdm_imag_patt_b.pattern}")
    read_rect_matrix_fast(file, rdm[..., 1])


MatrixLocation = namedtuple('MatrixLocation', ['idx', 'pos'])


def find_rdms(gdvlog: pathlib.Path, /, n_states: int | None = None) -> list[MatrixLocation]:
    locations: list[tuple[int, int]] = []
    with gdvlog.open(mode="r", encoding="utf8") as file_obj:
        with mmap.mmap(file_obj.fileno(), length=0, access=mmap.ACCESS_READ) as mmap_obj:
            finder = rdm_start_patt.finditer(mmap_obj)
            # if n_states is not None:
            #     iterator = zip(finder, range(n_states))
            # else:
            #     iterator = zip(finder, count(0))

            for m in tqdm(finder, desc='Searching for RDMs', total=None, unit='RDM'):
                idx = int(m['state']) - 1
                pos = int(m.span()[-1] + 1)
                locations.append((idx, pos))

    return locations


def batch_read_rdms(gdvlog: pathlib.Path, locations: Sequence[MatrixLocation], n_act_mo: int):
    n_rdms = len(locations)
    rdms = np.empty((n_rdms, n_act_mo, n_act_mo, 2), dtype=np.float_)
    with gdvlog.open(mode="r", encoding="utf8") as file_obj:
        with mmap.mmap(file_obj.fileno(), length=0, access=mmap.ACCESS_READ) as mmap_obj:
            for i, (idx, loc) in enumerate(locations):
                mmap_obj.seek(loc)
                read_rdm_fast(mmap_obj, rdms[i])
    return rdms.view(np.complex_).reshape(n_rdms, n_act_mo, n_act_mo)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('fast-parser')
    parser.add_argument('filename', type=pathlib.Path, help='Gaussian Log file')
    args = parser.parse_args()

    result = parse_gdvlog(args.filename, {'l910': [read_mc_spec,]})
    if 'n_mo_act' not in result:
        raise ValueError("Could not find number of active orbitals")

    pprint.pprint(result)

    rdm_locations = find_rdms(result['source'], n_states=result['n_configs'])
    pprint.pprint(rdm_locations)


