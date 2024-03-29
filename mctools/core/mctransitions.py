import warnings

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from .mcspace import MCSpace, MOSpacePartition
from .base import Consolidator, Selector
from .mcstates import MCStates
from .utils.constants import Eh2eV


__all__ = [
    'MCTransitions',
]


class MCTransitions(Consolidator):
    """Holds information about MCState overlaps: <I|Op|J> where, I and J are states, and Op some QM operator.

     Attributes:
        df: pd.DataFrame that holds information about the peaks;
        states: parent MCStates object, that holds information about states, associated with the peaks;

    Possible columns on df:
        SOURCE_COL: file from which the peak originates;
        INITIAL_STATE_COL: index of the initial state as defined in the `source`
        FINAL_STATE_COL: index of the final state as defined in the `source`

        OSC_COL: Oscillator strength, in arbitrary units, contributing to the peak;
        DE_COL: Energy difference between final and initial states, in Hartree, associated with the peak;

        INITIAL_COL: initial state index in `states.df.idx`, contributing to the peak;,
        FINAL_COL: final state index in `states.df.idx`, contributing to the peak;

        IDX_COLS: list of columns used to uniquely identify peak;
        DEFAULT_COLS: list of permanent columns on `df` for MCPeaks object to be valid;

    Future Development:
        TODO: use pd.MultiIndex to distinguish between different properties
        TODO: try using SOURCE_COL, INITIAL_STATE_COL, and FINAL_STATE_COL as index in df
    """
    INITIAL_COL = 'i'
    FINAL_COL = 'f'
    OSC_COL = 'osc'
    DE_COL = 'dE'
    RELATIVE_DE_COL = 'dE0'

    SOURCE_SUFFIX = 'state'
    INITIAL_STATE_COL = f'{INITIAL_COL}_{SOURCE_SUFFIX}'
    FINAL_STATE_COL = f'{FINAL_COL}_{SOURCE_SUFFIX}'

    SOURCE_COL = 'peak_source'

    IDX_COLS = [INITIAL_STATE_COL, FINAL_STATE_COL, SOURCE_COL]
    INIT_COLS = [INITIAL_STATE_COL, FINAL_STATE_COL, OSC_COL]
    DEFAULT_COLS = [INITIAL_STATE_COL, FINAL_STATE_COL, OSC_COL] + [SOURCE_COL, Consolidator.RESOURCE_COL]

    __slots__ = [
        'states',
        'tdms',
    ]

    tdms: npt.NDArray | None
    states: MCStates | None

    def __init__(self,
                 df: pd.DataFrame, /,
                 source: str = '',
                 states: MCStates | None = None,
                 tdms: npt.NDArray | None = None, *,
                 sort: bool = False,
                 keep_dark: bool = True) -> None:
        # TODO: implement validation of peaks against states.df
        super(MCTransitions, self).__init__(df, source=source, sort=False)

        # Validate and set MCStates, if provided
        self.states: MCStates | None = states
        if not self.are_states_set:
            warnings.warn("'states' attribute is not provided, expect reduced functionality")
        else:
            self.calculate_peak_energy(save=True, replace=True)

        # Validate and set TDMs, if provided
        self.tdms: npt.NDArray | None = tdms
        if self.tdms is not None:
            if self.tdms.ndim != 3:
                raise ValueError(f'Number of dimensions in TDMs array must be three: {tdms.ndim} != 3')

            if self.tdms.shape[-1] != tdms.shape[-2]:
                raise ValueError(f'Last two dimensions must be equal: {tdms.shape}')

            if self.are_states_set and self.space.n_mo_act != tdms.shape[-1]:
                raise ValueError(f'Last two dimensions must be equal to the size of active space: '
                                 f'{tdms.shape[-1]} != {self.space.n_mo_act}')

            if self.RESOURCE_COL not in self.df:
                if len(tdms) != len(self):
                    raise ValueError(f'Number of TDMs must be equal to the number of peaks: {len(tdms)} != {len(self)}')

                self._df[self.RESOURCE_COL] = np.arange(len(tdms))

        if not keep_dark:
            self.drop_dark()

        if sort:
            self.sort(col=self.DE_COL)

    def drop_dark(self) -> None:
        idx = self.filter(condition=lambda df: df[self.OSC_COL] > 0.0)
        self._df = self._df.loc[idx].copy(deep=True)

        self.tdms = self.tdms[self._df[self.RESOURCE_COL].values]
        self._df[self.RESOURCE_COL] = np.arange(len(self.tdms))

        self.reset_index()

    def analyze(self: 'MCTransitions', save: bool = True, replace: bool = False) -> pd.DataFrame | None:
        self.clear_properties()

        dfs = []
        try:
            dfs.append(self.calculate_peak_energy(save=save, replace=replace))
            dfs.append(self.calculate_rdm_diag_changes(save=save, replace=replace))
        except ValueError as err:
            warnings.warn(err.args[0])

        if not save:
            return pd.concat([self._df[self.OSC_COL], *dfs], axis=1)

    def contract_tdms(self, integral: npt.NDArray, spaces: MOSpacePartition) -> npt.NDArray:
        idx = self._df['resource_idx'].values

        int_as = integral[..., *spaces.active_2d]
        mel_as = np.einsum('...kl,jlk->j...', int_as, self.tdms[idx], optimize='optimal')  # matrix element active space

        int_is = integral[..., *spaces.inactive_2d]
        mel_is = int_is.trace(axis1=-1, axis2=-2)
        return mel_as + mel_is

    def get_state_properties(self: 'MCTransitions', props: list[str],
                             save: bool = True, replace: bool = False) -> pd.DataFrame | None:
        df_states = self.states.df[self.states.IDX_COLS + props]

        df = self._df[self.IDX_COLS].merge(
            df_states, how='left',
            right_on=self.states.IDX_COLS,
            left_on=[self.INITIAL_STATE_COL, self.SOURCE_COL],
            copy=True,
        ).drop(
            columns=self.states.IDX_COLS,
        ).rename(
            columns={col: f'{col}_{self.INITIAL_COL}' for col in props}
        ).merge(
            df_states, how='left',
            right_on=self.states.IDX_COLS,
            left_on=[self.FINAL_STATE_COL, self.SOURCE_COL],
            copy=True,
        ).drop(
            columns=self.states.IDX_COLS + self.IDX_COLS,
        ).rename(
            columns={col: f'{col}_{self.FINAL_COL}' for col in props}
        )

        if not save:
            return df

        self.update_properties(df, replace=replace)

    def calculate_state_idx(self: 'MCTransitions', save: bool = True, replace: bool = False) -> pd.DataFrame | None:
        df_states = self.states.df[self.states.IDX_COLS]

        df = self._df[self.IDX_COLS].merge(
            df_states, how='left',
            right_on=self.states.IDX_COLS,
            left_on=[self.INITIAL_STATE_COL, self.SOURCE_COL],
            copy=True,
        ).drop(
            columns=[self.states.SOURCE_COL],
        ).rename(
            columns={self.states.STATE_COL: self.INITIAL_COL}
        ).merge(
            df_states, how='left',
            right_on=self.states.IDX_COLS,
            left_on=[self.FINAL_STATE_COL, self.SOURCE_COL],
            copy=True,
        ).drop(
            columns=[self.states.SOURCE_COL] + self.IDX_COLS,
        ).rename(
            columns={self.states.STATE_COL: self.FINAL_COL}
        )

        if not save:
            return df

        self.update_properties(df, replace=replace)

    def calculate_peak_energy(self: 'MCTransitions', save: bool = True, replace: bool = False) -> pd.DataFrame | None:
        if not self.are_states_set:
            raise ValueError("'states' attribute must be set to calculate peak energy")

        df = self.get_state_properties([self.states.ENERGY_COL], save=False)

        # dE = E_f - E_i
        i_col = f'{self.states.ENERGY_COL}_{self.INITIAL_COL}'
        f_col = f'{self.states.ENERGY_COL}_{self.FINAL_COL}'
        df[self.DE_COL] = df[f_col] - df[i_col]
        df.drop(columns=[i_col, f_col], inplace=True)

        if not save:
            return df

        self.update_properties(df, replace=replace)

    def calculate_relative_energy(self: 'MCTransitions', scale: float = Eh2eV, offset: float = 0.0,
                                  idx: npt.ArrayLike | None = None, condition: Selector | None = None,
                                  save: bool = True, replace: bool = True,
                                  col_name: str = RELATIVE_DE_COL, **kwargs) -> pd.DataFrame | None:
        if self.DE_COL not in self.property_columns:
            raise ValueError(f'{self.DE_COL} is missing from properties, calculate it first')

        idx = self.filter(idx=idx, condition=condition)

        dE0 = self._df.loc[self._df.index[idx], self.DE_COL]
        dE0 *= scale
        dE0 += offset  # offset is in the units of Energy * Scale

        df = pd.DataFrame({col_name: dE0})

        if not save:
            return df

        self.update_properties(df, replace=replace)

    def calculate_rdm_diag_changes(self: 'MCTransitions', save: bool = True, replace: bool = False) -> pd.DataFrame | None:
        if not self.are_states_set:
            raise ValueError("'states' attribute must be set to calculate RDM diagonal differences")

        rdm_cols = list(self.space.mo_block_labels)
        if any(col not in self.states.df for col in rdm_cols):
            raise ValueError("states must have partitioned RDM diagonals")

        df = self.get_state_properties(rdm_cols, save=False)
        for col in rdm_cols:
            i_col = f'{col}_{self.INITIAL_COL}'
            f_col = f'{col}_{self.FINAL_COL}'
            df[col] = df[f'{col}_{self.FINAL_COL}'] - df[f'{col}_{self.INITIAL_COL}']
            df.drop(columns=[i_col, f_col], inplace=True)

        if not save:
            return df

        self.update_properties(df, replace=replace)

    # def extend(self, other: 'MCPeaks', **alignment_kwargs) -> NoReturn:
    #     raise NotImplementedError('Extension of spectrum is not yet implemented')
    #
    #     alignment = get_state_alignment(self.states, other.states, **alignment_kwargs)
    #     self.states.merge(other.states, alignment)
    #
    #     similar_peaks = self.find_similar(other, alignment=alignment)
    #     new_peaks = other._df[~other._df.index.isin(similar_peaks[f'{self.IDX_NAME}_right'])]
    #     df = pd.concat([self._df, new_peaks], axis=0, copy=True)
    #     df.drop(columns=[self.INITIAL_COL, self.FINAL_COL], inplace=True)
    #
    #     # Set new 'f' col
    #     df = df.merge(
    #         self.states._df[[self.states.STATE_COL, self.states.SOURCE_COL]].reset_index(), how='left',
    #         left_on=[self.FINAL_STATE_COL, self.SOURCE_COL],
    #         right_on=[self.states.STATE_COL, self.states.SOURCE_COL],
    #         copy=True,
    #     )
    #     df.drop(columns=self.states.STATE_COL, inplace=True)
    #     df.rename(columns={self.states.IDX_NAME: self.FINAL_COL}, inplace=True)
    #
    #     # Set new 'i' col
    #     df = df.merge(
    #         self.states._df[[self.states.STATE_COL, self.states.SOURCE_COL]].reset_index(), how='left',
    #         left_on=[self.INITIAL_STATE_COL, self.SOURCE_COL],
    #         right_on=[self.states.STATE_COL, self.states.SOURCE_COL],
    #         copy=True,
    #     )
    #     df.drop(columns=self.states.STATE_COL, inplace=True)
    #     df.rename(columns={self.states.IDX_NAME: self.INITIAL_COL}, inplace=True)
    #
    #     # Since some of the other's are not transferred into self.states we use for mapped states to set i
    #     missed_idx = df[df[self.INITIAL_COL].isna()].index
    #     alignment = get_state_alignment(self.states, other.states, **alignment_kwargs)
    #     state_map = get_state_map_from_alignment(alignment, target='left')
    #
    #     state_map = pd.concat([
    #         other.states._df.loc[state_map.index, [other.states.STATE_COL, other.states.SOURCE_COL]],
    #         state_map,
    #     ], axis=1)
    #
    #     # Map missed states
    #     temp_df2 = df.loc[missed_idx, [self.INITIAL_STATE_COL, self.SOURCE_COL]].merge(
    #         state_map, how='left',
    #         left_on=[self.INITIAL_STATE_COL, self.SOURCE_COL],
    #         right_on=[self.states.STATE_COL, self.states.SOURCE_COL],
    #     )
    #
    #     # Set missed states
    #     df.loc[missed_idx, self.INITIAL_COL] = temp_df2['idx'].values
    #     df[self.INITIAL_COL] = df[self.INITIAL_COL].astype(df[self.FINAL_COL].dtype)
    #
    #     self._df = df
    #
    #     self.sort()

    # def find_similar(self, other: 'MCPeaks', alignment: StateAlignment | None = None) -> pd.DataFrame:
    #     raise NotImplementedError('Finding similar peaks is not yet implemented')
    #
    #     alignment = get_state_alignment(self.states, other.states) if alignment is None else alignment
    #     state_map = get_state_map_from_alignment(alignment, target='left')
    #
    #     i_map = other.df_[other.INITIAL_COL].map(state_map, na_action='ignore')
    #     f_map = other.df_[other.FINAL_COL].map(state_map, na_action='ignore')
    #     other_mapped = pd.concat([i_map, f_map], axis=1)
    #
    #     df = self._df[[self.INITIAL_COL, self.FINAL_COL]].reset_index().merge(
    #         other_mapped.reset_index(),
    #         how='inner',
    #         left_on=[self.INITIAL_COL, self.FINAL_COL],
    #         right_on=[other.INITIAL_COL, other.FINAL_COL],
    #         suffixes=('_left', '_right')
    #     )
    #
    #     return df

    # def filter(self, i_cond: Selector | None = None, f_cond: Selector | None = None) -> pd.DataFrame:
    #     i_idx = self.states.filter(cond=i_cond, label_index=True)
    #     f_idx = self.states.filter(cond=f_cond, label_index=True)
    #     return self._df[self._df[self.INITIAL_COL].isin(i_idx) & self._df[self.FINAL_COL].isin(f_idx)]

    @classmethod
    def from_dict(cls, data: dict[str, Any], /,
                  df_key: str = 'df_peaks',
                  source_key: str = 'source',
                  states_key: str = 'states',
                  tdms_key: str = 'tdms',
                  instance_key: str = 'peaks',
                  **kwargs) -> 'MCTransitions':
        if isinstance(instance := data.get(instance_key, None), cls):
            return instance
        elif isinstance(instance, dict):
            instance = cls.from_dict(instance, **kwargs)
        elif instance is None:
            data.update(kwargs)

            df = data.pop(df_key)
            source = data.get(source_key, '')

            instance = data.setdefault(instance_key, cls(
                df, source,
                states=data.get(states_key, None),
                tdms=data.pop(tdms_key, None),
                keep_dark=kwargs.get('keep_dark', True),
                sort=kwargs.get('sort_peaks', True),
            ))
        else:
            raise ValueError(f"{cls.__name__} did not recognized '{instance_key}' "
                             f"item in data: {instance}")

        return instance

    # @classmethod
    # def from_spectra(cls, spectra: list['MCPeaks'], /, **kwargs) -> 'MCPeaks':
    #     spectra = sorted(spectra, key=lambda s: np.mean(s.energy_range))
    #
    #     base_spec = copy.deepcopy(spectra[0])
    #     for spec in spectra[1:]:
    #         base_spec.extend(spec, **kwargs)
    #
    #     return base_spec

    @property
    def E(self) -> np.ndarray:
        if self.DE_COL not in self._df:
            self.calculate_peak_energy(save=True)
        return self._df[self.DE_COL].values

    @property
    def min_energy(self) -> float:
        return float(self._df[self.DE_COL].min())

    @property
    def max_energy(self) -> float:
        return float(self._df[self.DE_COL].max())

    @property
    def energy_range(self) -> tuple[float, float]:
        return self.min_energy, self.max_energy

    @property
    def space(self) -> MCSpace:
        return self.states.mcspace

    @property
    def are_states_set(self) -> bool:
        return isinstance(self.states, MCStates)

    def __repr__(self) -> str:
        emin, emax = self.energy_range
        energy_str = f'E=[{emin:>11.6f}, {emax:>11.6f}]'
        return f'{self.__class__.__name__}({energy_str}, #peaks={len(self):>6,d})'

    def __len__(self) -> int:
        return len(self._df)
