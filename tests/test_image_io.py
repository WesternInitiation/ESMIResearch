from __future__ import annotations

import io
import tarfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from image_io import (
    HAS_RASTERIO,
    encode_reconstructed_image,
    list_archive_images,
    load_archive_image,
)

if HAS_RASTERIO:
    from rasterio.io import MemoryFile
    from rasterio.transform import from_origin


def _png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(np.arange(64, dtype=np.uint8).reshape(8, 8)).save(
        buffer, format="PNG"
    )
    return buffer.getvalue()


def _tar_bytes(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name, content in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    return buffer.getvalue()


class ArchiveImageTests(unittest.TestCase):
    def test_lists_and_loads_an_image_without_extracting_files(self) -> None:
        archive = _tar_bytes(
            {
                "metadata/readme.txt": b"test",
                "scene/source.png": _png_bytes(),
            }
        )

        self.assertEqual(list_archive_images(archive), ["scene/source.png"])
        loaded = load_archive_image(archive, "scene/source.png")

        self.assertEqual(loaded.band_order, ["gray"])
        self.assertEqual(loaded.bands["gray"].shape, (8, 8))
        self.assertEqual(loaded.metadata["archive_member"], "scene/source.png")

    def test_rejects_archives_without_supported_images(self) -> None:
        archive = _tar_bytes({"metadata/readme.txt": b"test"})

        with self.assertRaisesRegex(ValueError, "does not contain"):
            list_archive_images(archive)

    def test_rejects_archives_over_the_expanded_size_limit(self) -> None:
        archive = _tar_bytes({"source.png": _png_bytes()})

        with patch("image_io.MAX_ARCHIVE_EXPANDED_BYTES", 1):
            with self.assertRaisesRegex(ValueError, "expanded TAR archive"):
                list_archive_images(archive)

    def test_counts_non_regular_member_payloads_toward_archive_limit(self) -> None:
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w") as archive:
            special = tarfile.TarInfo("large-special-member")
            special.type = b"X"
            special.size = 64
            archive.addfile(special, io.BytesIO(b"x" * special.size))

            image = _png_bytes()
            regular = tarfile.TarInfo("source.png")
            regular.size = len(image)
            archive.addfile(regular, io.BytesIO(image))

        with patch("image_io.MAX_ARCHIVE_EXPANDED_BYTES", 32):
            with self.assertRaisesRegex(ValueError, "expanded TAR archive"):
                list_archive_images(buffer.getvalue())

    def test_malformed_archive_image_raises_an_image_error(self) -> None:
        archive = _tar_bytes({"source.png": b"not an image"})

        with self.assertRaises(OSError):
            load_archive_image(archive, "source.png")

    def test_exports_reconstructed_png(self) -> None:
        loaded = load_archive_image(
            _tar_bytes({"source.png": _png_bytes()}),
            "source.png",
        )

        output, filename, mime = encode_reconstructed_image(
            loaded, {"gray": loaded.bands["gray"]}
        )

        self.assertEqual(filename, "compressed_image.png")
        self.assertEqual(mime, "image/png")
        self.assertEqual(Image.open(io.BytesIO(output)).size, (8, 8))

    @unittest.skipUnless(HAS_RASTERIO, "rasterio is not installed")
    def test_geotiff_export_preserves_band_count_and_georeferencing(self) -> None:
        source = io.BytesIO()
        transform = from_origin(100.0, 200.0, 10.0, 10.0)
        profile = {
            "driver": "GTiff",
            "height": 8,
            "width": 8,
            "count": 2,
            "dtype": "uint16",
            "crs": "EPSG:4326",
            "transform": transform,
        }
        with MemoryFile() as memfile:
            with memfile.open(**profile) as dataset:
                dataset.write(np.ones((2, 8, 8), dtype=np.uint16))
                dataset.set_band_description(1, "B4")
                dataset.set_band_description(2, "B8")
                dataset.scales = (0.01, 0.02)
                dataset.offsets = (1.0, 2.0)
                dataset.units = ("reflectance", "reflectance")
                dataset.update_tags(PRODUCT="test-scene")
                dataset.update_tags(1, BAND_ROLE="red")
                mask = np.full((8, 8), 255, dtype=np.uint8)
                mask[0, 0] = 0
                dataset.write_mask(mask)
            source.write(memfile.read())

        loaded = load_archive_image(
            _tar_bytes({"scene/source.tif": source.getvalue()}),
            "scene/source.tif",
        )
        output, filename, mime = encode_reconstructed_image(
            loaded, loaded.bands
        )

        self.assertEqual(loaded.band_order, ["red", "nir"])
        self.assertEqual(filename, "compressed_image.tif")
        self.assertEqual(mime, "image/tiff")
        with MemoryFile(output) as memfile:
            with memfile.open() as dataset:
                self.assertEqual(dataset.count, 2)
                self.assertEqual(dataset.crs.to_string(), "EPSG:4326")
                self.assertEqual(dataset.transform, transform)
                self.assertEqual(dataset.compression.name.lower(), "deflate")
                self.assertEqual(dataset.descriptions, ("B4", "B8"))
                self.assertEqual(dataset.scales, (0.01, 0.02))
                self.assertEqual(dataset.offsets, (1.0, 2.0))
                self.assertEqual(dataset.units, ("reflectance", "reflectance"))
                self.assertEqual(dataset.tags()["PRODUCT"], "test-scene")
                self.assertEqual(dataset.tags(1)["BAND_ROLE"], "red")
                self.assertEqual(dataset.dataset_mask()[0, 0], 0)


if __name__ == "__main__":
    unittest.main()
