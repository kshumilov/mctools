from typing import Any, NoReturn

import pandas as pd

from .mcstates import MCStates, align_states, Selector

__all__ = [
    'MCSpectrum'
]


class MCSpectrum:
    """Holds information about multiconfigurational states.

     Attributes:
        df: Energy (in Eh) of states, spin, whether state is ground, and any other information that can be associated
            with a particular state. Indexed by state id starting from 0. df.shape = (#States, #Properties).
            df must contain the following columns:
                - i: int --- index of the initial state;
                - f: int --- index of the final state;
                - origin: str --- name of the file from which the peak originates;

        states: MCStates object
    """
    __slots__ = [
        'df',
        'states',
    ]

    INITIAL_COL = 'i'
    FINAL_COL = 'f'
    OSC_COL = 'osc'
    DE_COL = 'dE'

    SOURCE_SUFFIX = 'state'
    INITIAL_STATE_COL = f'{INITIAL_COL}_{SOURCE_SUFFIX}'
    FINAL_STATE_COL = f'{FINAL_COL}_{SOURCE_SUFFIX}'

    SOURCE_COL = 'source'

    COLS = [INITIAL_STATE_COL, FINAL_STATE_COL, SOURCE_COL, OSC_COL]

    IDX_NAME = 'idx'

    df: pd.DataFrame
    states: MCStates

    def __init__(self, peaks: pd.DataFrame, states: MCStates, source: str = '', keep_bright: bool = True):
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

        if keep_bright:
            self.df = self.df[self.df[self.OSC_COL] > 0.0].copy()
            self.reset_index()

        self.calculate_peak_energy()

    def analyze(self) -> NoReturn:
        self.states.analyze()

    def calculate_peak_energy(self, save: bool = True, replace: bool = True) -> pd.DataFrame | None:
        e_i_col = f'{self.states.E_COL}_{self.INITIAL_COL}'
        e_f_col = f'{self.states.E_COL}_{self.FINAL_COL}'

        df = self.df[[self.INITIAL_STATE_COL, self.FINAL_STATE_COL, self.SOURCE_COL, self.OSC_COL]].merge(
            self.states.df[self.states.COLS].reset_index(), how='left',
            right_on=[self.states.STATE_COL, self.states.SOURCE_COL],
            left_on=[self.INITIAL_STATE_COL, self.SOURCE_COL], copy=True,
        ).drop(columns=self.states.STATE_COL
        ).rename(columns={self.states.E_COL: e_i_col, self.states.IDX_NAME: self.INITIAL_COL})

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

    def filter(self, i_cond: Selector | None = None, f_cond: Selector | None = None) -> pd.DataFrame:
        i_idx = self.states.filter(cond=i_cond, label_index=True)
        f_idx = self.states.filter(cond=f_cond, label_index=True)
        return self.df[self.df[self.INITIAL_COL].isin(i_idx) & self.df[self.FINAL_COL].isin(f_idx)]

    def reset_index(self) -> NoReturn:
        self.df.reset_index(drop=True, inplace=True)
        self.df.index.name = self.IDX_NAME

    def clear_properties(self) -> NoReturn:
        curr_cols = set(self.df.columns) - set(self.COLS)
        self.df.drop(columns=curr_cols, inplace=True)

    def update_properties(self, new_df: pd.DataFrame, replace: bool = False) -> NoReturn:
        cols = set(new_df.columns)
        curr_cols = set(self.df.columns) - set(self.COLS)
        previous_cols = cols & curr_cols

        if replace:
            self.df.drop(columns=previous_cols, inplace=True)
            previous_cols.clear()

        new_cols = cols - previous_cols
        self.df = pd.concat([self.df, new_df[list(new_cols)]], axis=1, copy=False)
        self.df.update(new_df[list(previous_cols)])

    def merge(self, other: 'MCSpectrum') -> 'MCSpectrum':
        raise NotImplementedError('Merging MCSpectra is not yet implemented')

    @classmethod
    def from_dict(cls, data: dict[str, Any], **kwargs) -> 'MCSpectrum':
        data.update(kwargs)

        peaks = data.pop('df_peaks')
        source = data.get('source')
        if (states := data.pop('states', None)) is None:
            states = MCStates.from_dict(data)

        return cls(
            peaks, states, source=source,
            keep_bright=kwargs.get('keep_bright', True)
        )


def align_spectra(spec1: MCSpectrum, spec2: MCSpectrum):
    pass


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
