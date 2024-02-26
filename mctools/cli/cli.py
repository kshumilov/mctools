import click
from rich_click import RichCommand, RichGroup

from .parse import parse


@click.group(
    cls=RichGroup,
    name='mctools',
    help='Command line interface to mctools package'
)
@click.option(
    '--debug/--no-debug',
    show_default=True,
    help="Show the debug log messages",
)
def cli(debug):
    pass


cli.add_command(parse)


if __name__ == '__main__':
    cli()
