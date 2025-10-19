import os
import shutil
import tempfile
from enum import Enum
from pathlib import Path

from simppl.simple_pipeline import SimplePipeline
from ugbio_core.exec_utils import print_and_execute
from ugbio_core.logger import logger


class IntervalFileType(Enum):
    """Enumeration for supported interval file types."""

    BED = ".bed"
    INTERVAL_LIST = ".interval_list"


class IntervalFileError(Exception):
    """Custom exception for IntervalFile related errors."""


class TempFileManager:
    """Manages temporary files and directories for IntervalFile operations."""

    def __init__(self, *, scratchdir: bool | str = False):
        """
        Initialize the temporary file manager.

        Parameters
        ----------
        scratchdir : Union[bool, str], optional
            Temporary directory settings. If True, creates a new temp directory.
            If str, uses the provided path. If False, no temp directory is used.
            Default is False.
        """
        self.files_to_clean = []
        self.tmpdir = self._setup_temp_directory(scratchdir=scratchdir)

    def _setup_temp_directory(self, *, scratchdir: bool | str) -> str | None:
        """
        Set up temporary directory based on scratchdir parameter.

        Parameters
        ----------
        scratchdir : Union[bool, str]
            Temporary directory configuration parameter.

        Returns
        -------
        Optional[str]
            Path to temporary directory or None if not configured.
        """
        if scratchdir is True:
            tmpdir = tempfile.mkdtemp()
            self.files_to_clean.append(tmpdir)
            return tmpdir
        elif isinstance(scratchdir, str):
            return scratchdir
        return None

    def copy_to_temp(self, source_path: str) -> str:
        """
        Copy a file to the temporary directory if one exists.

        Parameters
        ----------
        source_path : str
            Path to the source file to copy.

        Returns
        -------
        str
            Path to the copied file in temp directory, or original path if no temp directory.
        """
        if self.tmpdir is None:
            return source_path

        dest_path = os.path.join(self.tmpdir, os.path.basename(source_path))
        if not os.path.exists(dest_path):
            shutil.copyfile(source_path, dest_path)
            self.add_file_to_clean(dest_path)
        return dest_path

    def add_file_to_clean(self, file_path: str):
        """
        Add a file to the cleanup list.

        Parameters
        ----------
        file_path : str
            Path to file that should be cleaned up later.
        """
        if file_path not in self.files_to_clean:
            self.files_to_clean.append(file_path)

    def cleanup(self):
        """Clean up all temporary files and directories."""
        for file_path in self.files_to_clean:
            logger.info(f"Cleaning up temporary file/directory: {file_path}")
            path = Path(file_path)
            try:
                if path.is_file():
                    os.remove(file_path)
            except OSError as e:
                logger.warning(f"Failed to cleanup {file_path}: {e}")
                raise RuntimeError(f"Failed to cleanup {file_path}: {e}") from e
            try:
                if path.is_dir():
                    shutil.rmtree(file_path)
            except OSError as e:
                logger.warning(f"Failed to cleanup {file_path}: {e}")
                raise RuntimeError(f"Failed to cleanup {file_path}: {e}") from e


class IntervalFileConverter:
    """Handles conversion between BED and interval list formats."""

    def __init__(self, sp: SimplePipeline | None = None):
        """
        Initialize the interval file converter.

        Parameters
        ----------
        sp : Optional[SimplePipeline], optional
            SimplePipeline instance for command execution. Default is None.
        """
        self.sp = sp

    def _execute_command(self, command: str, output_file: str | None = None):
        """
        Execute a command using the pipeline.

        Parameters
        ----------
        command : str
            Command to execute.
        output_file : Optional[str], optional
            Output file for command results. Default is None.
        """
        print_and_execute(command, output_file=output_file, simple_pipeline=self.sp, module_name=__name__)

    def bed_to_interval_list(self, bed_path: str, ref_dict: str) -> str:
        """
        Convert BED file to interval list format.

        Parameters
        ----------
        bed_path : str
            Path to the input BED file.
        ref_dict : str
            Path to the reference dictionary file.

        Returns
        -------
        str
            Path to the created interval list file.
        """
        self._validate_ref_dict(ref_dict)

        base_name = os.path.splitext(bed_path)[0]
        interval_list_path = f"{base_name}.interval_list"

        if not os.path.exists(interval_list_path):
            command = f"picard BedToIntervalList " f"I={bed_path} " f"O={interval_list_path} " f"SD={ref_dict}"
            self._execute_command(command)

        return interval_list_path

    def interval_list_to_bed(self, interval_list_path: str) -> str:
        """
        Convert interval list file to BED format.

        Parameters
        ----------
        interval_list_path : str
            Path to the input interval list file.

        Returns
        -------
        str
            Path to the created BED file.
        """
        base_name = os.path.splitext(interval_list_path)[0]
        bed_path = f"{base_name}.bed"

        if not os.path.exists(bed_path):
            command = f"picard IntervalListToBed I={interval_list_path} O={bed_path}"
            self._execute_command(command)

        return bed_path

    def _validate_ref_dict(self, ref_dict: str):
        """
        Validate that the reference dictionary file exists.

        Parameters
        ----------
        ref_dict : str
            Path to the reference dictionary file to validate.

        Raises
        ------
        IntervalFileError
            If the reference dictionary file does not exist.
        """
        if not os.path.isfile(ref_dict):
            raise IntervalFileError(f"Reference dictionary file does not exist: {ref_dict}")


class IntervalFile:
    """
    A class for handling interval files in both BED and interval list formats.

    This class provides automatic conversion between BED and interval list formats,
    with support for temporary file management and cleanup.
    """

    def __init__(
        self,
        sp: SimplePipeline | None = None,
        interval: str | None = None,
        ref: str | None = None,
        ref_dict: str | None = None,
        *,
        scratchdir: bool | str = False,
    ):
        """
        Initialize IntervalFile with automatic format conversion.

        Parameters
        ----------
        sp : Optional[SimplePipeline], optional
            SimplePipeline instance for command execution. Default is None.
        interval : Optional[str], optional
            Path to interval file (.bed or .interval_list) or None. Default is None.
        ref : Optional[str], optional
            Path to reference genome file. Default is None.
        ref_dict : Optional[str], optional
            Path to reference dictionary file (auto-deduced from ref if None). Default is None.
        scratchdir : Union[bool, str], optional
            Temporary directory settings (True, False, or path string). Default is False.
        """
        self.sp = sp
        self.temp_manager = TempFileManager(scratchdir=scratchdir)
        self.converter = IntervalFileConverter(sp)

        # Initialize state
        self._is_none = True
        self._bed_file_name: str | None = None
        self._interval_list_file_name: str | None = None

        if interval is not None:
            try:
                self._process_interval_file(interval, ref, ref_dict)
            except IntervalFileError as e:
                logger.error(str(e))
                # Maintain backward compatibility by setting error state
                self._is_none = True
                self._bed_file_name = None
                self._interval_list_file_name = None

    def _process_interval_file(self, interval: str, ref: str | None, ref_dict: str | None):
        """
        Process the input interval file and create both formats.

        Parameters
        ----------
        interval : str
            Path to the interval file to process.
        ref : Optional[str]
            Path to reference genome file.
        ref_dict : Optional[str]
            Path to reference dictionary file.
        """
        file_type = self._detect_file_type(interval)

        if file_type == IntervalFileType.BED:
            self._process_bed_file(interval, ref, ref_dict)
        elif file_type == IntervalFileType.INTERVAL_LIST:
            self._process_interval_list_file(interval)

        self._is_none = False

    def _detect_file_type(self, file_path: str) -> IntervalFileType:
        """
        Detect the file type based on file extension.

        Parameters
        ----------
        file_path : str
            Path to the file to analyze.

        Returns
        -------
        IntervalFileType
            The detected file type.

        Raises
        ------
        IntervalFileError
            If the file type is not supported.
        """
        path = Path(file_path)

        if path.suffix == IntervalFileType.BED.value:
            return IntervalFileType.BED
        elif path.suffix == IntervalFileType.INTERVAL_LIST.value:
            return IntervalFileType.INTERVAL_LIST
        else:
            raise IntervalFileError("the cmp_intervals should be of type interval list or bed")

    def _process_bed_file(self, bed_path: str, ref: str | None, ref_dict: str | None):
        """
        Process BED file and create interval list.

        Parameters
        ----------
        bed_path : str
            Path to the BED file to process.
        ref : Optional[str]
            Path to reference genome file.
        ref_dict : Optional[str]
            Path to reference dictionary file.
        """
        # Copy to temp directory if needed
        self._bed_file_name = self.temp_manager.copy_to_temp(bed_path)

        # Determine reference dictionary
        resolved_ref_dict = self._resolve_ref_dict(ref, ref_dict)

        # Convert to interval list
        interval_list_path = self.converter.bed_to_interval_list(self._bed_file_name, resolved_ref_dict)
        self._interval_list_file_name = interval_list_path

        # Add to cleanup if it was created
        if interval_list_path != bed_path.replace(".bed", ".interval_list"):
            self.temp_manager.add_file_to_clean(interval_list_path)

    def _process_interval_list_file(self, interval_list_path: str):
        """
        Process interval list file and create BED.

        Parameters
        ----------
        interval_list_path : str
            Path to the interval list file to process.
        """
        # Copy to temp directory if needed
        self._interval_list_file_name = self.temp_manager.copy_to_temp(interval_list_path)

        # Convert to BED
        bed_path = self.converter.interval_list_to_bed(self._interval_list_file_name)
        self._bed_file_name = bed_path

        # Add to cleanup if it was created
        if bed_path != interval_list_path.replace(".interval_list", ".bed"):
            self.temp_manager.add_file_to_clean(bed_path)

    def _resolve_ref_dict(self, ref: str | None, ref_dict: str | None) -> str:
        """
        Resolve the reference dictionary path.

        Parameters
        ----------
        ref : Optional[str]
            Path to reference genome file.
        ref_dict : Optional[str]
            Path to reference dictionary file.

        Returns
        -------
        str
            Path to the reference dictionary file.

        Raises
        ------
        IntervalFileError
            If neither ref_dict nor ref is provided.
        """
        if ref_dict is not None:
            return ref_dict

        if ref is None:
            raise IntervalFileError("Either ref_dict or ref must be provided for BED file conversion")

        return f"{ref}.dict"

    def as_bed_file(self) -> str | None:
        """
        Return the path to the BED file.

        Returns
        -------
        Optional[str]
            Path to the BED file or None if not available.
        """
        return self._bed_file_name

    def as_interval_list_file(self) -> str | None:
        """
        Return the path to the interval list file.

        Returns
        -------
        Optional[str]
            Path to the interval list file or None if not available.
        """
        return self._interval_list_file_name

    def is_none(self) -> bool:
        """
        Check if the interval file is None (not initialized).

        Returns
        -------
        bool
            True if the interval file was not properly initialized, False otherwise.
        """
        return self._is_none

    def __del__(self):
        """Clean up temporary files when the object is destroyed."""
        if hasattr(self, "temp_manager"):
            self.temp_manager.cleanup()

    def __enter__(self):
        """
        Context manager entry.

        Returns
        -------
        IntervalFile
            The IntervalFile instance for use in context manager.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit with cleanup.

        Parameters
        ----------
        exc_type : type
            Exception type if an exception occurred.
        exc_val : Exception
            Exception value if an exception occurred.
        exc_tb : traceback
            Exception traceback if an exception occurred.
        """
        self.temp_manager.cleanup()
