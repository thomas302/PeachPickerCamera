from typing import ClassVar, overload

class Config:
    """The main class of the multi/single-stage model config scheme (multi- stage
    models consists of interconnected single-stage models).

    @type config_version: str @ivar config_version: String representing config
    schema version in format 'x.y' where x is major version and y is minor version
    @type model: Model @ivar model: A Model object representing the neural network
    used in the archive."""
    configVersion: str | None
    model: Model
    @overload
    def __init__(self) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: depthai.nn_archive.v1.Config) -> None

        2. __init__(self: depthai.nn_archive.v1.Config, configVersion: str, model: depthai.nn_archive.v1.Model) -> None
        """
    @overload
    def __init__(self, configVersion: str, model: Model) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: depthai.nn_archive.v1.Config) -> None

        2. __init__(self: depthai.nn_archive.v1.Config, configVersion: str, model: depthai.nn_archive.v1.Model) -> None
        """

class DataType:
    """Data type of the input data (e.g., 'float32').

    Represents all existing data types used in i/o streams of the model.

    Precision of the model weights.

    Data type of the output data (e.g., 'float32').

    Members:

      BOOLEAN

      FLOAT16

      FLOAT32

      FLOAT64

      INT4

      INT8

      INT16

      INT32

      INT64

      UINT4

      UINT8

      UINT16

      UINT32

      UINT64

      STRING"""
    __members__: ClassVar[dict] = ...  # read-only
    BOOLEAN: ClassVar[DataType] = ...
    FLOAT16: ClassVar[DataType] = ...
    FLOAT32: ClassVar[DataType] = ...
    FLOAT64: ClassVar[DataType] = ...
    INT16: ClassVar[DataType] = ...
    INT32: ClassVar[DataType] = ...
    INT4: ClassVar[DataType] = ...
    INT64: ClassVar[DataType] = ...
    INT8: ClassVar[DataType] = ...
    STRING: ClassVar[DataType] = ...
    UINT16: ClassVar[DataType] = ...
    UINT32: ClassVar[DataType] = ...
    UINT4: ClassVar[DataType] = ...
    UINT64: ClassVar[DataType] = ...
    UINT8: ClassVar[DataType] = ...
    __entries: ClassVar[dict] = ...
    def __init__(self, value: int) -> None:
        """__init__(self: depthai.nn_archive.v1.DataType, value: int) -> None"""
    def __eq__(self, other: object) -> bool:
        """__eq__(self: object, other: object) -> bool"""
    def __hash__(self) -> int:
        """__hash__(self: object) -> int"""
    def __index__(self) -> int:
        """__index__(self: depthai.nn_archive.v1.DataType) -> int"""
    def __int__(self) -> int:
        """__int__(self: depthai.nn_archive.v1.DataType) -> int"""
    def __ne__(self, other: object) -> bool:
        """__ne__(self: object, other: object) -> bool"""
    @property
    def name(self) -> str:
        """name(self: object) -> str

        name(self: object) -> str
        """
    @property
    def value(self) -> int:
        """(arg0: depthai.nn_archive.v1.DataType) -> int"""

class Head:
    """Represents head of a model.

    @type name: str | None @ivar name: Optional name of the head. @type parser: str
    @ivar parser: Name of the parser responsible for processing the models output.
    @type outputs: List[str] | None @ivar outputs: Specify which outputs are fed
    into the parser. If None, all outputs are fed. @type metadata: C{HeadMetadata} |
    C{HeadObjectDetectionMetadata} | C{HeadClassificationMetadata} |
    C{HeadObjectDetectionSSDMetadata} | C{HeadSegmentationMetadata} |
    C{HeadYOLOMetadata} @ivar metadata: Metadata of the parser."""
    metadata: Metadata
    name: str | None
    outputs: list[str] | None
    parser: str
    def __init__(self) -> None:
        """__init__(self: depthai.nn_archive.v1.Head) -> None"""

class Input:
    """Represents input stream of a model.

    @type name: str @ivar name: Name of the input layer.

    @type dtype: DataType @ivar dtype: Data type of the input data (e.g.,
    'float32').

    @type input_type: InputType @ivar input_type: Type of input data (e.g.,
    'image').

    @type shape: list @ivar shape: Shape of the input data as a list of integers
    (e.g. [H,W], [H,W,C], [N,H,W,C], ...).

    @type layout: str @ivar layout: Lettercode interpretation of the input data
    dimensions (e.g., 'NCHW').

    @type preprocessing: PreprocessingBlock @ivar preprocessing: Preprocessing steps
    applied to the input data."""
    dtype: DataType
    inputType: InputType
    layout: str | None
    name: str
    preprocessing: PreprocessingBlock
    shape: list[int]
    def __init__(self) -> None:
        """__init__(self: depthai.nn_archive.v1.Input) -> None"""

class InputType:
    """

    Members:

      IMAGE

      RAW"""
    __members__: ClassVar[dict] = ...  # read-only
    IMAGE: ClassVar[InputType] = ...
    RAW: ClassVar[InputType] = ...
    __entries: ClassVar[dict] = ...
    def __init__(self, value: int) -> None:
        """__init__(self: depthai.nn_archive.v1.InputType, value: int) -> None"""
    def __eq__(self, other: object) -> bool:
        """__eq__(self: object, other: object) -> bool"""
    def __hash__(self) -> int:
        """__hash__(self: object) -> int"""
    def __index__(self) -> int:
        """__index__(self: depthai.nn_archive.v1.InputType) -> int"""
    def __int__(self) -> int:
        """__int__(self: depthai.nn_archive.v1.InputType) -> int"""
    def __ne__(self, other: object) -> bool:
        """__ne__(self: object, other: object) -> bool"""
    @property
    def name(self) -> str:
        """name(self: object) -> str

        name(self: object) -> str
        """
    @property
    def value(self) -> int:
        """(arg0: depthai.nn_archive.v1.InputType) -> int"""

class Metadata:
    """Metadata of the parser.

    Metadata for the object detection head.

    @type classes: list @ivar classes: Names of object classes detected by the
    model. @type n_classes: int @ivar n_classes: Number of object classes detected
    by the model. @type iou_threshold: float @ivar iou_threshold: Non-max supression
    threshold limiting boxes intersection. @type conf_threshold: float @ivar
    conf_threshold: Confidence score threshold above which a detected object is
    considered valid. @type max_det: int @ivar max_det: Maximum detections per
    image. @type anchors: list @ivar anchors: Predefined bounding boxes of different
    sizes and aspect ratios. The innermost lists are length 2 tuples of box sizes.
    The middle lists are anchors for each output. The outmost lists go from smallest
    to largest output.

    Metadata for the classification head.

    @type classes: list @ivar classes: Names of object classes classified by the
    model. @type n_classes: int @ivar n_classes: Number of object classes classified
    by the model. @type is_softmax: bool @ivar is_softmax: True, if output is
    already softmaxed

    Metadata for the SSD object detection head.

    @type boxes_outputs: str @ivar boxes_outputs: Output name corresponding to
    predicted bounding box coordinates. @type scores_outputs: str @ivar
    scores_outputs: Output name corresponding to predicted bounding box confidence
    scores.

    Metadata for the segmentation head.

    @type classes: list @ivar classes: Names of object classes segmented by the
    model. @type n_classes: int @ivar n_classes: Number of object classes segmented
    by the model. @type is_softmax: bool @ivar is_softmax: True, if output is
    already softmaxed

    Metadata for the YOLO head.

    @type yolo_outputs: list @ivar yolo_outputs: A list of output names for each of
    the different YOLO grid sizes. @type mask_outputs: list | None @ivar
    mask_outputs: A list of output names for each mask output. @type protos_outputs:
    str | None @ivar protos_outputs: Output name for the protos. @type
    keypoints_outputs: list | None @ivar keypoints_outputs: A list of output names
    for the keypoints. @type angles_outputs: list | None @ivar angles_outputs: A
    list of output names for the angles. @type subtype: str @ivar subtype: YOLO
    family decoding subtype (e.g. yolov5, yolov6, yolov7 etc.) @type n_prototypes:
    int | None @ivar n_prototypes: Number of prototypes per bbox in YOLO instance
    segmnetation. @type n_keypoints: int | None @ivar n_keypoints: Number of
    keypoints per bbox in YOLO keypoint detection. @type is_softmax: bool | None
    @ivar is_softmax: True, if output is already softmaxed in YOLO instance
    segmentation

    Metadata for the basic head. It allows you to specify additional fields.

    @type postprocessor_path: str | None @ivar postprocessor_path: Path to the
    postprocessor."""
    anchors: list[list[list[float]]] | None
    anglesOutputs: list[str] | None
    boxesOutputs: str | None
    classes: list[str] | None
    confThreshold: float | None
    extraParams: json
    iouThreshold: float | None
    isSoftmax: bool | None
    keypointsOutputs: list[str] | None
    maskOutputs: list[str] | None
    maxDet: int | None
    nClasses: int | None
    nKeypoints: int | None
    nPrototypes: int | None
    postprocessorPath: str | None
    protosOutputs: str | None
    scoresOutputs: str | None
    subtype: str | None
    yoloOutputs: list[str] | None
    def __init__(self) -> None:
        """__init__(self: depthai.nn_archive.v1.Metadata) -> None"""

class MetadataClass:
    """Metadata object defining the model metadata.

    Represents metadata of a model.

    @type name: str @ivar name: Name of the model. @type path: str @ivar path:
    Relative path to the model executable."""
    name: str
    path: str
    precision: DataType | None
    def __init__(self) -> None:
        """__init__(self: depthai.nn_archive.v1.MetadataClass) -> None"""

class Model:
    """A Model object representing the neural network used in the archive.

    Class defining a single-stage model config scheme.

    @type metadata: Metadata @ivar metadata: Metadata object defining the model
    metadata. @type inputs: list @ivar inputs: List of Input objects defining the
    model inputs. @type outputs: list @ivar outputs: List of Output objects defining
    the model outputs. @type heads: list @ivar heads: List of Head objects defining
    the model heads. If not defined, we assume a raw output."""
    heads: list[Head] | None
    inputs: list[Input]
    metadata: MetadataClass
    outputs: list[Output]
    def __init__(self) -> None:
        """__init__(self: depthai.nn_archive.v1.Model) -> None"""

class Output:
    """Represents output stream of a model.

    @type name: str @ivar name: Name of the output layer. @type dtype: DataType
    @ivar dtype: Data type of the output data (e.g., 'float32')."""
    dtype: DataType
    layout: str | None
    name: str
    shape: list[int] | None
    def __init__(self) -> None:
        """__init__(self: depthai.nn_archive.v1.Output) -> None"""

class PreprocessingBlock:
    """Preprocessing steps applied to the input data.

    Represents preprocessing operations applied to the input data.

    @type mean: list | None @ivar mean: Mean values in channel order. Order depends
    on the order in which the model was trained on. @type scale: list | None @ivar
    scale: Standardization values in channel order. Order depends on the order in
    which the model was trained on. @type reverse_channels: bool | None @ivar
    reverse_channels: If True input to the model is RGB else BGR. @type
    interleaved_to_planar: bool | None @ivar interleaved_to_planar: If True input to
    the model is interleaved (NHWC) else planar (NCHW). @type dai_type: str | None
    @ivar dai_type: DepthAI input type which is read by DepthAI to automatically
    setup the pipeline."""
    daiType: str | None
    interleavedToPlanar: bool | None
    mean: list[float] | None
    reverseChannels: bool | None
    scale: list[float] | None
    def __init__(self) -> None:
        """__init__(self: depthai.nn_archive.v1.PreprocessingBlock) -> None"""
