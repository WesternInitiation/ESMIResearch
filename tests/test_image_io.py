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
    list_archive_listing,
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

    def test_skips_junk_and_non_images_without_failing(self) -> None:
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w") as tar:
            for name, content in {
                "__MACOSX/._ignored.png": b"x",
                ".DS_Store": b"junk",
                "readme.txt": b"hi",
                "ok.png": _png_bytes(),
            }.items():
                info = tarfile.TarInfo(name)
                info.size = len(content)
                tar.addfile(info, io.BytesIO(content))
            # second same-named image — listing should keep the first only
            dup = _png_bytes()
            info = tarfile.TarInfo("ok.png")
            info.size = len(dup)
            tar.addfile(info, io.BytesIO(dup))

        self.assertEqual(list_archive_images(buffer.getvalue()), ["ok.png"])

    def test_lists_images_even_when_archive_has_large_non_image_members(self) -> None:
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w") as archive:
            special = tarfile.TarInfo("nested/dir/")
            special.type = tarfile.DIRTYPE
            special.size = 0
            archive.addfile(special)

            big = tarfile.TarInfo("blob.bin")
            big.size = 10_000
            archive.addfile(big, io.BytesIO(b"z" * big.size))

            image = _png_bytes()
            regular = tarfile.TarInfo("source.png")
            regular.size = len(image)
            archive.addfile(regular, io.BytesIO(image))

        self.assertEqual(list_archive_images(buffer.getvalue()), ["source.png"])

    def test_lists_folders_inside_nested_archive(self) -> None:
        archive = _tar_bytes(
            {
                "scene/a/band4.tif": _png_bytes(),
                "scene/a/band5.tif": _png_bytes(),
                "scene/b/band4.tif": _png_bytes(),
                "readme.txt": b"meta",
            }
        )
        listing = list_archive_listing(archive)
        self.assertEqual(
            listing["images"],
            ["scene/a/band4.tif", "scene/a/band5.tif", "scene/b/band4.tif"],
        )
        self.assertEqual(listing["folders"], ["scene", "scene/a", "scene/b"])

    def test_lists_gzipped_tar_via_magic_bytes(self) -> None:
        raw = _tar_bytes({"scene/source.png": _png_bytes()})
        gz_buf = io.BytesIO()
        import gzip

        with gzip.GzipFile(fileobj=gz_buf, mode="wb") as gz:
            gz.write(raw)
        listing = list_archive_listing(gz_buf.getvalue(), filename="upload.tar")
        self.assertEqual(listing["images"], ["scene/source.png"])
        self.assertEqual(listing["folders"], ["scene"])

    def test_skips_oversized_images_during_list_instead_of_failing(self) -> None:
        png = _png_bytes()
        archive = _tar_bytes(
            {
                "huge.tif": b"x" * (len(png) + 50),
                "source.png": png,
            }
        )
        with patch("image_io.MAX_ARCHIVE_IMAGE_BYTES", len(png) + 10):
            self.assertEqual(list_archive_images(archive), ["source.png"])

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
