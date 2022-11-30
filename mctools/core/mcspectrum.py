import copy
import warnings
from typing import Any, NoReturn

import numpy as np
import pandas as pd

from .mcspace import MCSpace
from .mcstates import MCStates, Selector
from .utils import get_state_alignment, get_state_map_from_alignment, StateAlignment

__all__ = [
    'MCSpectrum'
]


class MCSpectrum:
    """Holds information about multiconfigurational peaks.

     Attributes:
        df: Energy (in Eh) of states, spin, whether state is ground, and any other information that can be associated
            with a particular state. Indexed by state id starting from 0. df.shape = (#States, #Properties).
            df must contain the following columns:
                - i: int --- index of the initial state;
                - f: int --- index of the final state;
                - origin: str --- name of the file from which the peak originates;

        states: MCStates object

    TODO: use pd.MultiIndex to distinguish between different properties
    TODO: try using SOURCE_COL, INITIAL_STATE_COL, and FINAL_STATE_COL as index in df
    """
    __slots__ = [
        'df',
        'states',
        'COLS',
    ]

    INITIAL_COL = 'i'
    FINAL_COL = 'f'
    OSC_COL = 'osc'
    DE_COL = 'dE'

    SOURCE_SUFFIX = 'state'
    INITIAL_STATE_COL = f'{INITIAL_COL}_{SOURCE_SUFFIX}'
    FINAL_STATE_COL = f'{FINAL_COL}_{SOURCE_SUFFIX}'

    SOURCE_COL = 'source'

    IDX_COLS = [INITIAL_STATE_COL, FINAL_STATE_COL, SOURCE_COL]
    DEFAULT_COLS = [INITIAL_STATE_COL, FINAL_STATE_COL, SOURCE_COL, OSC_COL]
    DATA_COLS = [INITIAL_COL, FINAL_COL, OSC_COL, DE_COL]

    IDX_NAME = 'idx'

    df: pd.DataFrame
    states: MCStates

    def __init__(self, peaks: pd.DataFrame, states: MCStates, source: str = '',
                 keep_dark: bool = False, sort_peaks: bool = False):
        if self.INITIAL_STATE_COL not in peaks:
            raise ValueError(f'peaks must have {self.INITIAL_STATE_COL}')

        if self.FINAL_STATE_COL not in peaks:
            raise ValueError(f'peaks must have {self.FINAL_STATE_COL}')

        if self.OSC_COL not in peaks:
            raise ValueError(f'peaks must have {self.OSC_COL}')

        if self.SOURCE_COL not in peaks:
            if source:
                peaks[self.SOURCE_COL] = source
            else:
                raise ValueError(f'either states must have {self.SOURCE_COL} or source argument must be passed to '
                                 f'{self.__class__.__name__}')

        # TODO: implement validation of peaks against states.df
        self.df = peaks
        self.states = states

        if not keep_dark:
            self.df = self.df[self.df[self.OSC_COL] > 0.0].copy()
            self.reset_index()

        self._calculate_peak_energy()
        self.COLS = list(self.df.columns)

        if sort_peaks:
            self.sort()

    def analyze(self) -> NoReturn:
        self.states.analyze()

    def _calculate_peak_energy(self, save: bool = True, replace: bool = True) -> pd.DataFrame | None:
        if self.DE_COL in self.df:
            warnings.warn('Peak energy has already been calculated')
            save = False

        e_i_col = f'{self.states.E_COL}_{self.INITIAL_COL}'
        e_f_col = f'{self.states.E_COL}_{self.FINAL_COL}'

        # Find 'i' column and 'E_i' for each peak
        df = self.df[self.DEFAULT_COLS].merge(
            self.states.df[self.states.COLS].reset_index(), how='left',
            right_on=[self.states.STATE_COL, self.states.SOURCE_COL],
            left_on=[self.INITIAL_STATE_COL, self.SOURCE_COL], copy=True,
        ).drop(columns=self.states.STATE_COL
        ).rename(columns={self.states.E_COL: e_i_col, self.states.IDX_NAME: self.INITIAL_COL})

        # Calculate 'f' column and 'E_f'
        df = df.merge(
            self.states.df[self.states.COLS].reset_index(), how='left',
            right_on=[self.states.STATE_COL, self.states.SOURCE_COL],
            left_on=[self.FINAL_STATE_COL, self.SOURCE_COL], copy=True,
        ).drop(columns=self.states.STATE_COL
        ).rename(columns={self.states.E_COL: e_f_col, self.states.IDX_NAME: self.FINAL_COL})

        df[self.DE_COL] = (df[e_f_col] - df[e_i_col])
        df.drop(columns=[e_f_col, e_i_col,
                         self.INITIAL_STATE_COL, self.FINAL_STATE_COL, self.SOURCE_COL, self.OSC_COL], inplace=True)

        if not save:
            return df

        self.update_properties(df, replace=replace)

    def extend(self, other: 'MCSpectrum', **alignment_kwargs) -> NoReturn:
        alignment = get_state_alignment(self.states, other.states, **alignment_kwargs)
        self.states.merge(other.states, alignment)

        similar_peaks = self.find_similar(other, alignment=alignment)
        new_peaks = other.df[~other.df.index.isin(similar_peaks[f'{self.IDX_NAME}_right'])]
        df = pd.concat([self.df, new_peaks], axis=0, copy=True)
        df.drop(columns=[self.INITIAL_COL, self.FINAL_COL], inplace=True)

        # Set new 'f' col
        df = df.merge(
            self.states.df[[self.states.STATE_COL, self.states.SOURCE_COL]].reset_index(), how='left',
            left_on=[self.FINAL_STATE_COL, self.SOURCE_COL],
            right_on=[self.states.STATE_COL, self.states.SOURCE_COL],
            copy=True,
        )
        df.drop(columns=self.states.STATE_COL, inplace=True)
        df.rename(columns={self.states.IDX_NAME: self.FINAL_COL}, inplace=True)

        # Set new 'i' col
        df = df.merge(
            self.states.df[[self.states.STATE_COL, self.states.SOURCE_COL]].reset_index(), how='left',
            left_on=[self.INITIAL_STATE_COL, self.SOURCE_COL],
            right_on=[self.states.STATE_COL, self.states.SOURCE_COL],
            copy=True,
        )
        df.drop(columns=self.states.STATE_COL, inplace=True)
        df.rename(columns={self.states.IDX_NAME: self.INITIAL_COL}, inplace=True)

        # Since some of the other's are not transferred into self.states we use for mapped states to set i
        missed_idx = df[df[self.INITIAL_COL].isna()].index
        alignment = get_state_alignment(self.states, other.states, **alignment_kwargs)
        state_map = get_state_map_from_alignment(alignment, target='left')

        state_map = pd.concat([
            other.states.df.loc[state_map.index, [other.states.STATE_COL, other.states.SOURCE_COL]],
            state_map,
        ], axis=1)

        # Map missed states
        temp_df2 = df.loc[missed_idx, [self.INITIAL_STATE_COL, self.SOURCE_COL]].merge(
            state_map, how='left',
            left_on=[self.INITIAL_STATE_COL, self.SOURCE_COL],
            right_on=[self.states.STATE_COL, self.states.SOURCE_COL],
        )

        # Set missed states
        df.loc[missed_idx, self.INITIAL_COL] = temp_df2['idx'].values
        df[self.INITIAL_COL] = df[self.INITIAL_COL].astype(df[self.FINAL_COL].dtype)

        self.df = df

        self.sort()

    def find_similar(self, other: 'MCSpectrum', alignment: StateAlignment | None = None) -> pd.DataFrame:
        alignment = get_state_alignment(self.states, other.states) if alignment is None else alignment
        state_map = get_state_map_from_alignment(alignment, target='left')

        i_map = other.df[other.INITIAL_COL].map(state_map, na_action='ignore')
        f_map = other.df[other.FINAL_COL].map(state_map, na_action='ignore')
        other_mapped = pd.concat([i_map, f_map], axis=1)

        df = self.df[[self.INITIAL_COL, self.FINAL_COL]].reset_index().merge(
            other_mapped.reset_index(),
            how='inner',
            left_on=[self.INITIAL_COL, self.FINAL_COL],
            right_on=[other.INITIAL_COL, other.FINAL_COL],
            suffixes=('_left', '_right')
        )

        return df

    def sort(self, col: str = DE_COL) -> NoReturn:
        if col in self.df:
            self.df.sort_values(col, ignore_index=True, inplace=True)

        self.reset_index()

    def filter(self, i_cond: Selector | None = None, f_cond: Selector | None = None) -> pd.DataFrame:
        i_idx = self.states.filter(cond=i_cond, label_index=True)
        f_idx = self.states.filter(cond=f_cond, label_index=True)
        return self.df[self.df[self.INITIAL_COL].isin(i_idx) & self.df[self.FINAL_COL].isin(f_idx)]

    def reset_index(self) -> NoReturn:
        self.df.reset_index(drop=True, inplace=True)
        self.df.index.name = self.IDX_NAME

    def clear_properties(self) -> NoReturn:
        curr_cols = set(self.df.columns) - set(self.DEFAULT_COLS)
        self.df.drop(columns=curr_cols, inplace=True)

    def update_properties(self, new_df: pd.DataFrame, replace: bool = False) -> NoReturn:
        cols = set(new_df.columns)
        curr_cols = set(self.df.columns) - set(self.DEFAULT_COLS)
        previous_cols = cols & curr_cols

        if replace:
            self.df.drop(columns=previous_cols, inplace=True)
            previous_cols.clear()

        new_cols = cols - previous_cols
        self.df = pd.concat([self.df, new_df[list(new_cols)]], axis=1, copy=False)
        self.df.update(new_df[list(previous_cols)])

    @classmethod
    def from_dict(cls, data: dict[str, Any], /, **kwargs) -> 'MCSpectrum':
        data.update(kwargs)

        peaks = data.pop('df_peaks')
        source = data.get('source')
        if (states := data.pop('states', None)) is None:
            states = MCStates.from_dict(data)

        return cls(
            peaks, states, source=source,
            keep_dark=kwargs.get('keep_dark', False),
            sort_peaks=kwargs.get('sort_peaks', True),
        )

    @classmethod
    def from_spectra(cls, spectra: list['MCSpectrum'], /, **kwargs) -> 'MCSpectrum':
        spectra = sorted(spectra, key=lambda s: np.mean(s.energy_range))

        base_spec = copy.deepcopy(spectra[0])
        for spec in spectra[1:]:
            base_spec.extend(spec, **kwargs)

        return base_spec

    @property
    def min_energy(self) -> np.float64:
        return self.df[self.DE_COL].min()

    @property
    def max_energy(self) -> np.float64:
        return self.df[self.DE_COL].max()

    @property
    def energy_range(self) -> tuple[np.float64, np.float64]:
        return self.min_energy, self.max_energy

    @property
    def space(self) -> MCSpace:
        return self.states.space

    def __repr__(self) -> str:
        emin, emax = self.energy_range
        energy_str = f'E=[{emin:>11.6f}, {emax:>11.6f}]'
        return f'{self.__class__.__name__}({energy_str}, #peaks={len(self):>6,d})'

    def __len__(self) -> int:
        return len(self.df)


if __name__ == '__main__':
    import os
    from mctools.core.mcspace import MCSpace
    from mctools.parser.gaussian.utils import parse_gdvlog
    from mctools.parser.gaussian.l910 import l910_parser_funcs

    data_dir = os.path.join('..', '..', 'data')
    gdvlog = os.path.join(data_dir, 'rasci_1.log')

    space = MCSpace.from_json(os.path.join(data_dir, 'rasci_1.space.json'))
    data = parse_gdvlog(gdvlog, l910_parser_funcs, n_ground=14)
    spectrum = MCSpectrum.from_dict(data, space=space)
    spectrum.analyze()
