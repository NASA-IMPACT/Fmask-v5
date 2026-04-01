"""Commands for managing Fmask ancillary data (models and auxiliary rasters).

The expected layout on disk mirrors the SharePoint distribution ZIP:

    <project-root>/
        data/
            global_gt30.tif
            global_gt30_slope.tif
            global_gt30_aspect.tif
            global_gswo150.tif
        model/
            lightgbm_cloud_l8.pk
            lightgbm_cloud_l8_sample.pk
            lightgbm_cloud_l7.pk
            lightgbm_cloud_l7_sample.pk
            lightgbm_cloud_s2.pk
            lightgbm_cloud_s2_sample.pk
            unet_cloud_l8.pt
            unet_cloud_l7.pt
            unet_cloud_s2.pt
        src/
            ...

Use ``fmask-data install`` to extract ``data/`` and ``model/`` from the
downloaded ZIP, and ``fmask-data pack`` to recreate a compatible ZIP for
redistribution.
"""

import zipfile
from pathlib import Path

import click

# src/fmask/cli/ -> src/fmask/ -> src/ -> <project-root>
_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

_DATA_DIRS = ("data", "model")


@click.group()
def main():
    """Manage Fmask ancillary data files."""


@main.command()
@click.argument("zip_path", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--dest",
    type=click.Path(file_okay=False),
    default=None,
    help="Root directory to extract into. Defaults to the project root.",
)
def install(zip_path, dest):
    """Extract data/ and model/ from the Fmask distribution ZIP.

    ZIP_PATH is the path to the ZIP file downloaded from the Fmask SharePoint.
    """
    root = Path(dest) if dest else _PROJECT_ROOT
    root.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as zf:
        members = zf.namelist()

        # The ZIP has a single top-level directory (e.g. "fmask/").
        # Detect it from the first entry.
        prefix = members[0].split("/")[0] + "/"

        targets = [f"{prefix}{d}/" for d in _DATA_DIRS]
        to_extract = [m for m in members if any(m.startswith(t) for t in targets)]

        if not to_extract:
            raise click.ClickException(
                f"No data/ or model/ directories found under '{prefix}' in the ZIP. "
                "Is this the correct Fmask distribution file?"
            )

        for member in to_extract:
            # Strip the top-level prefix so files land at root/data/... and root/model/...
            rel = member[len(prefix):]
            if not rel:
                continue
            dest_path = root / rel
            if member.endswith("/"):
                dest_path.mkdir(parents=True, exist_ok=True)
            else:
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                dest_path.write_bytes(zf.read(member))

        click.echo(f"Installed data/ and model/ to {root}")


@main.command()
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False),
    default="fmask_data.zip",
    show_default=True,
    help="Path for the output ZIP file.",
)
@click.option(
    "--source",
    type=click.Path(exists=True, file_okay=False),
    default=None,
    help="Root directory containing data/ and model/. Defaults to the project root.",
)
def pack(output, source):
    """Create a ZIP of data/ and model/ for redistribution.

    The resulting ZIP matches the layout of the Fmask SharePoint distribution,
    with both directories nested under a top-level ``fmask/`` prefix.
    """
    root = Path(source) if source else _PROJECT_ROOT

    missing = [d for d in _DATA_DIRS if not (root / d).is_dir()]
    if missing:
        raise click.ClickException(
            f"Missing directories under {root}: {', '.join(missing)}. "
            "Run 'fmask-data install' first, or check --source."
        )

    output_path = Path(output)
    file_count = 0
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for dir_name in _DATA_DIRS:
            src_dir = root / dir_name
            for file in sorted(src_dir.rglob("*")):
                if file.is_file():
                    arcname = f"fmask/{dir_name}/{file.relative_to(src_dir)}"
                    zf.write(file, arcname)
                    file_count += 1

    click.echo(f"Packed {file_count} files into {output_path}")
