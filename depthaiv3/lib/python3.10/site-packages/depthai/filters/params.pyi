from _typeshed import Incomplete
from typing import ClassVar

class MedianFilter:
    """Members:

      MEDIAN_OFF

      KERNEL_3x3

      KERNEL_5x5

      KERNEL_7x7"""
    __members__: ClassVar[dict] = ...  # read-only
    KERNEL_3x3: ClassVar[MedianFilter] = ...
    KERNEL_5x5: ClassVar[MedianFilter] = ...
    KERNEL_7x7: ClassVar[MedianFilter] = ...
    MEDIAN_OFF: ClassVar[MedianFilter] = ...
    __entries: ClassVar[dict] = ...
    def __init__(self, value: int) -> None:
        """__init__(self: depthai.filters.params.MedianFilter, value: int) -> None"""
    def __eq__(self, other: object) -> bool:
        """__eq__(self: object, other: object) -> bool"""
    def __hash__(self) -> int:
        """__hash__(self: object) -> int"""
    def __index__(self) -> int:
        """__index__(self: depthai.filters.params.MedianFilter) -> int"""
    def __int__(self) -> int:
        """__int__(self: depthai.filters.params.MedianFilter) -> int"""
    def __ne__(self, other: object) -> bool:
        """__ne__(self: object, other: object) -> bool"""
    @property
    def name(self) -> str:
        """name(self: object) -> str

        name(self: object) -> str
        """
    @property
    def value(self) -> int:
        """(arg0: depthai.filters.params.MedianFilter) -> int"""

class SpatialFilter:
    alpha: float
    delta: int
    enable: bool
    holeFillingRadius: int
    numIterations: int
    def __init__(self) -> None:
        """__init__(self: depthai.filters.params.SpatialFilter) -> None"""

class SpeckleFilter:
    differenceThreshold: int
    enable: bool
    speckleRange: int
    def __init__(self) -> None:
        """__init__(self: depthai.filters.params.SpeckleFilter) -> None"""

class TemporalFilter:
    """Temporal filtering with optional persistence."""

    class PersistencyMode:
        """Persistency algorithm type.

        Members:

          PERSISTENCY_OFF : 

          VALID_8_OUT_OF_8 : 

          VALID_2_IN_LAST_3 : 

          VALID_2_IN_LAST_4 : 

          VALID_2_OUT_OF_8 : 

          VALID_1_IN_LAST_2 : 

          VALID_1_IN_LAST_5 : 

          VALID_1_IN_LAST_8 : 

          PERSISTENCY_INDEFINITELY : """
        __members__: ClassVar[dict] = ...  # read-only
        PERSISTENCY_INDEFINITELY: ClassVar[TemporalFilter.PersistencyMode] = ...
        PERSISTENCY_OFF: ClassVar[TemporalFilter.PersistencyMode] = ...
        VALID_1_IN_LAST_2: ClassVar[TemporalFilter.PersistencyMode] = ...
        VALID_1_IN_LAST_5: ClassVar[TemporalFilter.PersistencyMode] = ...
        VALID_1_IN_LAST_8: ClassVar[TemporalFilter.PersistencyMode] = ...
        VALID_2_IN_LAST_3: ClassVar[TemporalFilter.PersistencyMode] = ...
        VALID_2_IN_LAST_4: ClassVar[TemporalFilter.PersistencyMode] = ...
        VALID_2_OUT_OF_8: ClassVar[TemporalFilter.PersistencyMode] = ...
        VALID_8_OUT_OF_8: ClassVar[TemporalFilter.PersistencyMode] = ...
        __entries: ClassVar[dict] = ...
        def __init__(self, value: int) -> None:
            """__init__(self: depthai.filters.params.TemporalFilter.PersistencyMode, value: int) -> None"""
        def __eq__(self, other: object) -> bool:
            """__eq__(self: object, other: object) -> bool"""
        def __hash__(self) -> int:
            """__hash__(self: object) -> int"""
        def __index__(self) -> int:
            """__index__(self: depthai.filters.params.TemporalFilter.PersistencyMode) -> int"""
        def __int__(self) -> int:
            """__int__(self: depthai.filters.params.TemporalFilter.PersistencyMode) -> int"""
        def __ne__(self, other: object) -> bool:
            """__ne__(self: object, other: object) -> bool"""
        @property
        def name(self) -> str:
            """name(self: object) -> str

            name(self: object) -> str
            """
        @property
        def value(self) -> int:
            """(arg0: depthai.filters.params.TemporalFilter.PersistencyMode) -> int"""
    alpha: float
    delta: int
    enable: bool
    persistencyMode: Incomplete
    def __init__(self) -> None:
        """__init__(self: depthai.filters.params.TemporalFilter) -> None"""
