"""Console script for nanofinderparser."""

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.progress import Progress
from rich.table import Table

from nanofinderparser import load_mdt, load_mdt_images, load_smd
from nanofinderparser.units import Units
from nanofinderparser.utils import SaveMapCoords

# Using "Optional" as typer doesn't accept "X | Y" notation

app = typer.Typer(
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
)
console = Console()


@app.command(
    "convert",
    short_help="Convert a SMD file(s) to CSV.",
    no_args_is_help=True,
)
def convert_smd(
    input_path: Annotated[Path, typer.Argument(..., help="Path to the SMD file or folder")],
    output: Annotated[Path | None, typer.Argument(help="Output folder for CSV file(s)")] = None,
    units: Annotated[
        Units, typer.Option(case_sensitive=False, help="Units for the spectral axis")
    ] = Units.raman_shift,
    save_mapcoords: Annotated[
        SaveMapCoords,
        typer.Option(..., case_sensitive=False, help="How to save mapping coordinates"),
    ] = SaveMapCoords.combined,
) -> None:
    """Convert SMD file(s) to CSV format.

    If input is a folder, converts all SMD files in the folder.
    """
    if input_path.is_file():
        files_to_convert = [input_path]
    elif input_path.is_dir():
        files_to_convert = list(input_path.glob("*.smd"))
    else:
        msg = "Input must be a file or directory"
        raise typer.BadParameter(msg)

    if not files_to_convert:
        console.print("[yellow]No SMD files found in the specified directory.[/yellow]")
        return

    output_dir = output or input_path.parent if input_path.is_file() else input_path
    output_dir.mkdir(parents=True, exist_ok=True)

    with Progress() as progress:
        task = progress.add_task("[cyan]Converting files...", total=len(files_to_convert))
        try:
            for file in files_to_convert:
                mapping = load_smd(file)
                output_file = output_dir / file.with_suffix(".csv").name

                mapping.to_csv(
                    path=output_dir,
                    filename=output_file.name,
                    spectral_units=units.value,
                    save_mapcoords=save_mapcoords.value,
                )
                progress.update(task, advance=1)
                console.print(f"[green]Converted {file} to {output_file}[/green]")
            console.print(f"[green]Successfully converted {len(files_to_convert)} file(s)[/green]")
        except Exception as e:
            console.print(f"[red]Error converting '{file}': {e}[/red]")
            raise typer.Exit(code=1) from e


@app.command(
    no_args_is_help=True,
)
def info(file: Annotated[Path, typer.Argument(..., help="Path to the SMD file")]) -> None:
    """Display information about an SMD file."""
    try:
        mapping = load_smd(file)
        table = Table(title=f"SMD File Information: {file.name}")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="magenta")

        table.add_row("Date", str(mapping.datetime))
        table.add_row("Laser Wavelength", f"{mapping.laser_wavelength:.2f} nm")
        table.add_row("Laser Power", f"{mapping.laser_power:.2f} mW")
        exposure_time = mapping.get_exposure_time()
        if exposure_time is not None:
            table.add_row("Exposure Time", f"{exposure_time:.2f} s")
        else:
            table.add_row("Exposure Time", "Not available")
        table.add_row(
            "CCD Temperature (°C)",
            f"{mapping.scanned_frame_parameters.data_calibration.channels[0].channel_info.temperature}",
        )
        table.add_row(
            "Map Size",
            f"{mapping.map_size[0]:.2f} x {mapping.map_size[1]:.2f} {mapping.step_units[0]}",
        )
        table.add_row("Map Steps", f"{mapping.map_steps[0]} x {mapping.map_steps[1]}")
        table.add_row(
            "Step Size",
            f"{mapping.step_size[0]:.4f} x {mapping.step_size[1]:.4f} {mapping.step_units[0]}",
        )
        table.add_row("Spectral Points", str(mapping.get_spectral_axis_len()))
        table.add_row("Spectral Units", mapping._get_channel_axis_unit())  # noqa: SLF001

        console.print(table)
    except Exception as e:
        console.print(f"[red]Error reading file: {e}[/red]")
        raise typer.Exit(code=1) from e


@app.command(
    "convert-mdt",
    short_help="Convert a MDT file(s) of individual spectra to CSV.",
    no_args_is_help=True,
)
def convert_mdt(
    input_path: Annotated[Path, typer.Argument(..., help="Path to the MDT file or folder")],
    output: Annotated[Path | None, typer.Argument(help="Output folder for CSV file(s)")] = None,
    units: Annotated[
        Units, typer.Option(case_sensitive=False, help="Units for the spectral axis")
    ] = Units.raman_shift,
    combined: Annotated[
        bool, typer.Option(help="Write all the spectra of a file to a single CSV")
    ] = False,
    maps: Annotated[bool, typer.Option(help="Also export the 2-D maps stored in the file")] = False,
) -> None:
    """Convert MDT file(s) to CSV format.

    If input is a folder, converts all MDT files in the folder. Each spectrum is written to its
    own CSV unless --combined is given, since spectra in the same file may have been recorded
    over different spectral axes.
    """
    if input_path.is_file():
        files_to_convert = [input_path]
    elif input_path.is_dir():
        files_to_convert = list(input_path.glob("*.mdt"))
    else:
        msg = "Input must be a file or directory"
        raise typer.BadParameter(msg)

    if not files_to_convert:
        console.print("[yellow]No MDT files found in the specified directory.[/yellow]")
        return

    output_dir = output or (input_path.parent if input_path.is_file() else input_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    with Progress() as progress:
        task = progress.add_task("[cyan]Converting files...", total=len(files_to_convert))
        try:
            written = 0
            for file in files_to_convert:
                spectra = load_mdt(file)
                written += len(
                    spectra.to_csv(
                        path=output_dir,
                        filename=file.stem,
                        spectral_units=units.value,
                        combined=combined,
                    )
                )
                if maps:
                    written += len(load_mdt_images(file).to_csv(output_dir, filename=file.stem))
                progress.update(task, advance=1)
                console.print(f"[green]Converted {file}[/green]")
            console.print(
                f"[green]Successfully wrote {written} CSV file(s) "
                f"from {len(files_to_convert)} MDT file(s)[/green]"
            )
        except Exception as e:
            console.print(f"[red]Error converting '{file}': {e}[/red]")
            raise typer.Exit(code=1) from e


@app.command("info-mdt", no_args_is_help=True)
def info_mdt(file: Annotated[Path, typer.Argument(..., help="Path to the MDT file")]) -> None:
    """Display information about a MDT file."""
    try:
        spectra = load_mdt(file)
        images = load_mdt_images(file)

        table = Table(title=f"MDT File Information: {file.name}")
        table.add_column("Title", style="cyan")
        table.add_column("Kind", style="green")
        table.add_column("Size", style="magenta")
        table.add_column("Range", style="magenta")
        table.add_column("Date", style="white")

        for spectrum in spectra:
            axis = spectrum.spectral_axis
            table.add_row(
                spectrum.title,
                "spectrum",
                f"{spectrum.spectral_axis_len} pts",
                f"{axis[0]:.2f} - {axis[-1]:.2f} {spectrum.spectral_axis_unit}",
                str(spectrum.datetime or "unknown"),
            )
        for image in images:
            y_size, x_size = image.shape
            table.add_row(
                image.title,
                "map",
                f"{x_size} x {y_size}",
                f"{image.values.min():.3g} - {image.values.max():.3g} {image.value_unit}",
                str(image.datetime or "unknown"),
            )

        console.print(table)
        if spectra:
            wavelengths = {s.laser_wavelength for s in spectra if s.laser_wavelength is not None}
            if wavelengths:
                console.print(
                    "Laser wavelength: " + ", ".join(f"{wl:.2f} nm" for wl in sorted(wavelengths))
                )
    except Exception as e:
        console.print(f"[red]Error reading file: {e}[/red]")
        raise typer.Exit(code=1) from e


# ??? Is this needed? Conversion back to a SMD is not straightforward
@app.command()
def export_smd(
    mapping: Annotated[Path, typer.Argument(..., help="Path to the input CSV file")],
    output: Annotated[Path | None, typer.Argument(help="Output path for the SMD file")] = None,
) -> None:
    """Export a CSV file back to SMD format."""
    try:
        # TODO: Implement the logic to convert CSV back to SMD
        console.print("[yellow]Export to SMD functionality not yet implemented.[/yellow]")
    except Exception as e:
        console.print(f"[red]Error exporting to SMD: {e}[/red]")
        raise typer.Exit(code=1) from e


if __name__ == "__main__":
    app()
