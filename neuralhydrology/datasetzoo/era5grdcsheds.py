from pathlib import Path
from typing import List, Dict, Union, Optional

import pandas as pd
import numpy as np
import xarray

from neuralhydrology.datasetzoo.basedataset import BaseDataset
from neuralhydrology.utils.config import Config


class Era5GrdcSheds(BaseDataset):
    """Class encoding the dataset created through merging forcings from ERA5 Land, basins defined by HydroSHEDS and
    streamflow timeseries provided by GRDC.

    Parameters
    ----------
    cfg : Config
        The run configuration.
    is_train : bool
        Defines if the dataset is used for training or evaluating. If True (training), means/stds for each feature
        are computed and stored to the run directory. If one-hot encoding is used, the mapping for the one-hot encoding 
        is created and also stored to disk. If False, a `scaler` input is expected and similarly the `id_to_int` input
        if one-hot encoding is used. 
    period : {'train', 'validation', 'test'}
        Defines the period for which the data will be loaded
    basin : str, optional
        If passed, the data for only this basin will be loaded. Otherwise the basin(s) are read from the appropriate
        basin file, corresponding to the `period`.
    additional_features : List[Dict[str, pd.DataFrame]], optional
        List of dictionaries, mapping from a basin id to a pandas DataFrame. This DataFrame will be added to the data
        loaded from the dataset, and all columns are available as 'dynamic_inputs', 'evolving_attributes' and
        'target_variables'
    id_to_int : Dict[str, int], optional
        If the config argument 'use_basin_id_encoding' is True in the config and period is either 'validation' or 
        'test', this input is required. It is a dictionary, mapping from basin id to an integer (the one-hot encoding).
    scaler : Dict[str, Union[pd.Series, xarray.DataArray]], optional
        If period is either 'validation' or 'test', this input is required. It contains the centering and scaling
        for each feature and is stored to the run directory during training (train_data/train_data_scaler.yml).
    """

    def __init__(self,
                 cfg: Config,
                 is_train: bool,
                 period: str,
                 basin: str = None,
                 additional_features: List[Dict[str, pd.DataFrame]] = [],
                 id_to_int: Dict[str, int] = {},
                 scaler: Dict[str, Union[pd.Series, xarray.DataArray]] = {}):
        # initialize parent class
        super(Era5GrdcSheds, self).__init__(cfg=cfg,
                                              is_train=is_train,
                                              period=period,
                                              basin=basin,
                                              additional_features=additional_features,
                                              id_to_int=id_to_int,
                                              scaler=scaler)

    def _load_basin_data(self, basin: str) -> pd.DataFrame:
        """Load basin time series data
        
        This function is used to load the time series data (meteorological forcing, streamflow, etc.) and make available
        as time series input for model training later on. Make sure that the returned dataframe is time-indexed.
        
        Parameters
        ----------
        basin : str
            Basin identifier as string.

        Returns
        -------
        pd.DataFrame
            Time-indexed DataFrame, containing the time series data (e.g., forcings + discharge).
        """

        return load_era5_grdc_sheds_timeseries(data_dir=self.cfg.data_dir, basin=basin)

    def _load_attributes(self) -> pd.DataFrame:
        """Load dataset attributes
        
        This function is used to load basin attribute data (e.g. CAMELS catchments attributes) as a basin-indexed 
        dataframe with features in columns.
        
        Returns
        -------
        pd.DataFrame
            Basin-indexed DataFrame, containing the attributes as columns.
        """
        return load_era5_grdc_sheds_attributes(data_dir=self.cfg.data_dir)

def load_era5_grdc_sheds_timeseries(data_dir: Path, basin: str) -> pd.DataFrame:
    """Loads the timeseries data of one basin from the ERA5-GRDC-SHEDS dataset.
    
    Parameters
    ----------
    data_dir : Path
        Path to the root directory of ERA5-GRDC-SHEDS that has to include a sub-directory called 'timeseries'. This 
        sub-directory has to contain another sub-directory called either 'csv' or 'netcdf', depending on the choice 
        of the filetype argument. By default, netCDF files are loaded from the 'netcdf' subdirectory.
    basin : str
        The HydroSHEDS basin_id in the form basin_{HYBAS_ID}

    Raises
    ------
    FileNotFoundError
        If no timeseries file exists for the basin.
    """

    # Get HydroSHEDS level from basin ID
    lv = int(int(basin) / 10000000) % 10

    filepath = data_dir / "timeseries" / f"timeseries_lv0{lv}" / f"basin_{basin}.csv"

    if not filepath.is_file():
        raise FileNotFoundError(f"No basin file found at {filepath}.")


    df = pd.read_csv(filepath, parse_dates=['date'])
    df = df.set_index('date')

    # for feat in ['sro_sum', 'ssro_sum', 'streamflow']:
    #     df[feat] = np.log(df[feat]+0.001)

    return df

def load_era5_grdc_sheds_attributes(data_dir: Path,
                                    basins: Optional[List[str]] = None) -> pd.DataFrame:
    """Load the attributes of the ERA5-GRDC-SHEDS dataset.

    Parameters
    ----------
    data_dir : Path
        Path to the root directory of ERA5-GRDC-SHEDS that has to include a sub-directory called 'attributes' which contain the 
        attributes of all sub-datasets in separate folders.
    basins : List[str], optional
        If passed, returns only attributes for the basins specified in this list. Otherwise, the attributes of all 
        basins are returned.

    Raises
    ------
    ValueError
        If any of the requested basins does not exist in the attribute files or if both, basins and sub-dataset are 
        passed but at least one of the basins is not part of the corresponding sub-dataset.

    Returns
    -------
    pd.DataFrame
        A basin indexed DataFrame with all attributes as columns.
    """

    attributes_dir = data_dir / "attributes"

    subdataset_dirs = [d for d in (data_dir / "attributes").glob('*') if d.is_dir()]

    if basins:
        print("BASINS = TRUE")
        # Get list of unique sub datasets from the basin strings.
        subdataset_names = list(set("attributes_lv0" + str(int(int(basin) / 10000000) % 10) for basin in basins))

        # Check if all subdatasets exist.
        missing_subdatasets = [s for s in subdataset_names if not (data_dir / "attributes" / s).is_dir()]
        if missing_subdatasets:
            raise FileNotFoundError(f"Could not find subdataset directories for {missing_subdatasets}.")

        # Subset subdataset_dirs to only the required subsets.
        subdataset_dirs = [s for s in subdataset_dirs if s.name in subdataset_names]

    # Load all required attribute files.
    dfs = []
    for subdataset_dir in subdataset_dirs:
        dfs.append(_load_attribute_files_of_subdataset(subdataset_dir))

    # Merge all DataFrames along the basin index.
    df = pd.concat(dfs, axis=0)

    if basins:
        if any(b not in df.index for b in basins):
            raise ValueError('Some basins are missing static attributes.')
        # Subset to only the requested basins.
        df = df.loc[basins]

    # transform index in string
    df.index = [str(basin_id) for basin_id in df.index]

    return df

def _load_attribute_files_of_subdataset(subdataset_dir: Path) -> pd.DataFrame:
    """Loads all attribute files for one subdataset and merges them into one DataFrame."""
    dfs = []
    for csv_file in subdataset_dir.glob("*.csv"):
        dfs.append(pd.read_csv(csv_file, index_col="basin_id"))
    return pd.concat(dfs, axis=1)