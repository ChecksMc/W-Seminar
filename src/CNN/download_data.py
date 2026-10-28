"""Download and safely extract the EmoArt-5k data archive."""

import argparse
import shutil
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from . import param

REPO: str = "https://huggingface.co/datasets/printblue/EmoArt-5k/resolve/main"


@dataclass(frozen=True)
class DownloadArguments:
    """Command-line choices for retrieving and unpacking EmoArt data.

    Parameters:
        skip_images: Whether to download only annotation metadata.
        no_extract: Whether to leave the downloaded image archive compressed.

    Returns:
        An immutable download configuration object.
    """

    skip_images: bool
    no_extract: bool


def download(url: str, destination: Path) -> None:
    """Download a remote file through a temporary path and replace atomically.

    Parameters:
        url: HTTP or HTTPS address of the file.
        destination: Local path where the completed file should be stored.

    Returns:
        None. Progress is printed while bytes are received.

    Raises:
        ValueError: If the URL does not use HTTP or HTTPS.
        OSError: If network access or local file writing fails.
    """
    if not url.startswith(("https://", "http://")):
        raise ValueError(f"Only HTTP and HTTPS download URLs are supported: {url}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "W-Seminar-EmoArt-CNN/1.0"})
    partial_path = destination.with_suffix(destination.suffix + ".part")
    try:
        with urllib.request.urlopen(request) as response, partial_path.open("wb") as output:
            total_bytes = int(response.headers.get("Content-Length", 0))
            downloaded_bytes = 0
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
                downloaded_bytes += len(chunk)
                if total_bytes > 0:
                    progress = downloaded_bytes / total_bytes
                    print(f"\r{destination.name}: {progress:.1%}", end="", flush=True)
        if partial_path.stat().st_size == 0:
            raise OSError(f"Downloaded file is empty: {url}")
        partial_path.replace(destination)
    except Exception:
        partial_path.unlink(missing_ok=True)
        raise
    print()


def _safe_extract(archive_path: Path, destination: Path) -> None:
    """Extract a tar archive while rejecting paths that escape destination.

    Parameters:
        archive_path: Existing gzip-compressed tar file to extract.
        destination: Root directory allowed to receive extracted members.

    Returns:
        None. Extracted archive members are written below destination.

    Raises:
        FileNotFoundError: If the archive path does not exist.
        ValueError: If an archive member resolves outside destination.
        tarfile.TarError: If the archive is corrupt or has an unsupported format.
        OSError: If extraction cannot write to destination.
    """
    if not archive_path.is_file():
        raise FileNotFoundError(f"Image archive does not exist: {archive_path}")
    destination.mkdir(parents=True, exist_ok=True)
    safe_root = destination.resolve()
    with tarfile.open(archive_path, "r:gz") as bundle:
        members = bundle.getmembers()
        for member in members:
            resolved_member = (safe_root / member.name).resolve()
            if not resolved_member.is_relative_to(safe_root):
                raise ValueError(f"Archive member escapes the extraction directory: {member.name}")
            if member.issym() or member.islnk() or member.isdev():
                raise ValueError(f"Archive contains an unsupported link or device: {member.name}")
        bundle.extractall(destination, members=members)


def _parse_arguments() -> DownloadArguments:
    """Parse options for downloading annotations and artwork files.

    Parameters:
        None.

    Returns:
        Namespace with annotation-only and no-extraction flags.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-images", action="store_true", help="Download annotations only")
    parser.add_argument("--no-extract", action="store_true", help="Keep the image archive compressed")
    parsed = parser.parse_args()
    return DownloadArguments(skip_images=bool(parsed.skip_images), no_extract=bool(parsed.no_extract))


def main() -> None:
    """Download EmoArt files and extract images to the configured raw-data path.

    Parameters:
        None.

    Returns:
        None. Files are stored below ``data/raw/emoart5k``.

    Raises:
        OSError: If network access, file writing, or extraction fails.
        ValueError: If the downloaded archive contains unsafe paths.
        tarfile.TarError: If the downloaded archive is invalid.
    """
    arguments = _parse_arguments()
    image_archive = param.RAW_DIR / "Images.tar.gz"
    annotation_file = param.ANNOTATION_FILE
    if not arguments.skip_images and not image_archive.is_file():
        print("Downloading the approximately 2.6 GB EmoArt-5k image archive")
        download(f"{REPO}/Images.tar.gz", image_archive)
    if not annotation_file.is_file():
        download(f"{REPO}/annotation.json", annotation_file)
    if not arguments.skip_images and not arguments.no_extract and not param.IMAGE_DIR.is_dir():
        print(f"Extracting artwork images to {param.RAW_DIR}")
        _safe_extract(image_archive, param.RAW_DIR)
        alternate_directories = (
            param.RAW_DIR / "images",
            param.RAW_DIR / "EmoArt-5k" / "Images",
        )
        for alternate_directory in alternate_directories:
            if alternate_directory.is_dir():
                shutil.move(str(alternate_directory), str(param.IMAGE_DIR))
                break
        if not param.IMAGE_DIR.is_dir():
            raise FileNotFoundError(
                f"Extraction completed but expected image directory is missing: {param.IMAGE_DIR}"
            )
    print(f"EmoArt data is available under {param.RAW_DIR}")


if __name__ == "__main__":
    main()
