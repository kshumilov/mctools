import warnings

import numpy as np
import pandas as pd

from .mcstates import MCStates

StateAlignment = list[tuple[slice | None, slice | None]]


def get_state_alignment(left: MCStates, right: MCStates,
                        offset_left: int = 0, offset_right: int = 0,
                        margin: int = 3, tol=0.7, ignore_space: bool = False,
                        ignore_overlap: bool = False) -> StateAlignment:
    """Aligns states based on energy and overlap between CI vectors.

    Assumes that states are sorted and contain no duplicates
    """
    if not ignore_space and left.space != right.space:
        raise ValueError('Spaces are different CI vector overlap is poorly defined')

    if not len(left) and not len(right):
        return []
    elif not len(left) and len(right):
        return [(None, np.s_[offset_right:offset_right + len(right)])]
    elif len(left) and not len(right):
        return [(np.s_[offset_left:offset_left + len(left)], None)]

    # Swap such that 'left' comes before 'right'
    if swapped := (right.E[0] < left.E[0]):
        offset_left, offset_right = offset_right, offset_left
        left, right = right, left

    # check if states overlap
    if left.E[-1] < right.E[0] or left.E[0] > right.E[-1]:
        sl_left = np.s_[np.s_[offset_left:offset_left + len(left)]]
        sl_right = np.s_[np.s_[offset_right:offset_left + len(right)]]

        if swapped:
            sl_left, sl_right = sl_right, sl_left

        return [(sl_left, None), (None, sl_right)]

    result = []

    # Find an approximate location of right[first] in left within window the size of +/- margin
    w_begin = max(0, np.searchsorted(left.E, right.E[0]) - margin)
    w_end = min(w_begin + 2 * margin, len(left))

    # Find the best overlap in the window
    idx, overlap = left[w_begin:w_end].find_similar(right[0])
    if overlap[0] < tol:
        msg = f'Overlap below tolerance: {tol} > {overlap[0]}'
        if ignore_overlap:
            warnings.warn(msg)
        else:
            raise ValueError(msg)

    # Mark states that present in s1 but not s2
    start_left, start_right = idx[0] + w_begin, 0
    if start_left != start_right:
        sl_left = np.s_[offset_left:offset_left + start_left]
        if swapped:
            result.append((None, sl_left))
        else:
            result.append((sl_left, None))

    # Iterate through states one by one to make find the first non-overlapping states
    end_left, end_right = start_left + 1, start_right + 1
    while end_left < len(left) and end_right < len(right):
        v1 = left.ci_vecs[[end_left], :]
        v2 = right.ci_vecs[[end_right], :]

        overlap = np.abs((v1 @ v2.getH()).toarray())[0, 0]
        if overlap > tol:
            end_left += 1
            end_right += 1
        else:
            break

    # Create overlapping slices
    sl_left = np.s_[offset_left + start_left:offset_left + end_left]
    sl_right = np.s_[offset_left + start_right:offset_left + end_right]

    # Update the offsets
    offset_left += end_left
    offset_right += end_right

    if swapped:
        offset_left, offset_right = offset_right, offset_left
        sl_left, sl_right = sl_right, sl_left
        left, right = right, left

    result.append((sl_left, sl_right))

    # Check if there are states left and recursively search for the next overlap
    if end_left <= len(left) or end_right <= len(right):
        result.extend(get_state_alignment(left[end_left:], right[end_right:],
                                          offset_left=offset_left, offset_right=offset_right,
                                          margin=margin, tol=tol,
                                          ignore_space=ignore_space,
                                          ignore_overlap=ignore_overlap))
    return result


def get_state_map_from_alignment(alignment: StateAlignment, target='right') -> pd.Series:
    state_map: dict[int, int] = {}
    for region in alignment:
        match region:
            case slice(), slice():
                left, right = region
                if target == 'left':
                    right, left = left, right

                state_map.update({
                    local: target for local, target in
                    zip(range(left.start, left.stop), range(right.start, right.stop))
                })
            case _:
                continue

    return pd.Series(state_map, name='idx')
