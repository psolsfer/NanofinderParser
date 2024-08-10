"""Console script for nanofinderparser."""
import sys
import typer
app = typer.Typer()

@app.command()
def main(args=None) -> None:
    """Console script for nanofinderparser."""
    typer.echo("Replace this message by putting your code into "
               "nanofinderparser.cli.main")
    typer.echo("See Typer documentation at https://typer.tiangolo.com/")
    return None


if __name__ == "__main__":
    app()
