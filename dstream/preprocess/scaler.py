from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    MaxAbsScaler,
    Normalizer,
)
from typing import Tuple, Any, Type, Callable, Union, Optional
import pandas as pd
from dstream.preprocess.base import IDataScaler
from dstream.utils.logged import setLogging


logger = setLogging().getLogger("Scaler")


class DataScaler(IDataScaler):
    SCALERS: dict[str, Type] = {
        "standard": StandardScaler,
        "minmax": MinMaxScaler,
        "robust": RobustScaler,
        "maxabs": MaxAbsScaler,
        "normalize": Normalizer,
    }

    def __init__(
        self,
        method: Optional[str] = "standard",
        custom_scaler: Union[Type, Callable, None] = None,
        **kwargs,
    ):
        self.method = method.lower()
        self.kwargs = kwargs

        if custom_scaler is not None:
            logger.info("Using custom scaler.")
            self.scaler = (
                custom_scaler(**kwargs)
                if callable(custom_scaler)
                else custom_scaler
            )
        else:
            if self.method not in self.SCALERS:
                raise ValueError(
                    f"Unsupported scaler method '{self.method}'. "
                    f"Available options are: {list(self.SCALERS.keys())} "
                    f"or provide a 'custom_scaler' instance."
                )
            self.scaler = self.SCALERS[self.method](**kwargs)

    def scale(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Any]:
        logger.info(f"Scaling data using '{self.method}' scaler...")
        try:
            scaled = self.scaler.fit_transform(data)
            scaled_df = pd.DataFrame(scaled, columns=data.columns, index=data.index)
            logger.info("Data scaling complete.")
            return scaled_df, self.scaler
        except Exception as e:
            logger.error(f"Error during data scaling: {e}")
            raise
