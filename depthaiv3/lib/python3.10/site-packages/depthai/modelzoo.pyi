import os

def getDefaultCachePath() -> os.PathLike:
    """getDefaultCachePath() -> os.PathLike

    Get the default cache path (where models are cached)
    """
def getDefaultModelsPath() -> os.PathLike:
    """getDefaultModelsPath() -> os.PathLike

    Get the default models path (where yaml files are stored)
    """
def getDownloadEndpoint() -> str:
    """getDownloadEndpoint() -> str

    Get the download endpoint (for model querying)
    """
def getHealthEndpoint() -> str:
    """getHealthEndpoint() -> str

    Get the health endpoint (for internet check)
    """
def setDefaultCachePath(path: os.PathLike) -> None:
    """setDefaultCachePath(path: os.PathLike) -> None

    Set the default cache path (where models are cached)

    Parameter ``path``:
    """
def setDefaultModelsPath(path: os.PathLike) -> None:
    """setDefaultModelsPath(path: os.PathLike) -> None

    Set the default models path (where yaml files are stored)

    Parameter ``path``:
    """
def setDownloadEndpoint(endpoint: str) -> None:
    """setDownloadEndpoint(endpoint: str) -> None

    Set the download endpoint (for model querying)

    Parameter ``endpoint``:
    """
def setHealthEndpoint(endpoint: str) -> None:
    """setHealthEndpoint(endpoint: str) -> None

    Set the health endpoint (for internet check)

    Parameter ``endpoint``:
    """
