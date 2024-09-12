"""Console script for nanofinderparser."""

from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.table import Table

from nanofinderparser import load_smd
from nanofinderparser.units import Units
from nanofinderparser.utils import SaveMapCoords

# ruff: noqa: UP007 # Using "Optional" as typer doesn't accept "X | Y" notation

app = typer.Typer()
console = Console()


@app.command("convert", short_help="Convert a SMD file to CSV.")
def convert_smd(
    file: Annotated[Path, typer.Argument(..., help="Path to the SMD file")],
    output: Annotated[Optional[Path], typer.Argument(help="Output path for the CSV file")] = None,
    units: Annotated[
        Units, typer.Option(case_sensitive=False, help="Units for the spectral axis")
    ] = Units.raman_shift,
    save_mapcoords: Annotated[
        SaveMapCoords,
        typer.Option(..., case_sensitive=False, help="How to save mapping coordinates"),
    ] = SaveMapCoords.combined,
) -> None:
    """Convert an SMD file to CSV format."""
    try:
        mapping = load_smd(file)
        output = output or file.with_suffix(".csv")
        mapping.to_csv(
            path=output.parent,
            filename=output.name,
            spectral_units=units.value,  # type: ignore[arg-type]
            save_mapcoords=save_mapcoords.value,
        )
        console.print(f"[green]Successfully converted {file} to {output}[/green]")
    except Exception as e:
        console.print(f"[red]Error converting file: {e}[/red]")
        raise typer.Exit(code=1) from e


@app.command()
def info(file: Annotated[Path, typer.Argument(..., help="Path to the SMD file")]) -> None:
    """Display information about an SMD file."""
    try:
        mapping = load_smd(file)
        table = Table(title=f"SMD File Information: {file.name}")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="magenta")

        table.add_row("Date", str(mapping.date))
        table.add_row("Laser Wavelength", f"{mapping.laser_wavelength:.2f} nm")
        table.add_row("Laser Power", f"{mapping.laser_power:.2f} mW")
        table.add_row(
            "Map Size",
            f"{mapping.map_size[0]:.2f} x {mapping.map_size[1]:.2f} {mapping.step_units[0]}",
        )
        table.add_row("Map Steps", f"{mapping.map_steps[0]} x {mapping.map_steps[1]}")
        table.add_row("Spectral Points", str(mapping.get_spectral_axis_len()))
        table.add_row("Spectral Units", mapping._get_channel_axis_unit())  # noqa: SLF001

        console.print(table)
    except Exception as e:
        console.print(f"[red]Error reading file: {e}[/red]")
        raise typer.Exit(code=1) from e


# ??? Is this needed? Conversion back to a SMD is not straightforward
@app.command()
def export_smd(
    mapping: Annotated[Path, typer.Argument(..., help="Path to the input CSV file")],
    output: Annotated[Optional[Path], typer.Argument(help="Output path for the SMD file")] = None,
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
