import click

from .parser import parse


@click.group(name='mctools')
def main():
    pass


main.add_command(parse)


if __name__ == '__main__':
    main()
