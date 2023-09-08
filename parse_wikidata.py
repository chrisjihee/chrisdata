import logging

from chrisbase.data import AppTyper

logger = logging.getLogger(__name__)
app = AppTyper()


@app.command()
def parse():
    pass


if __name__ == "__main__":
    app()
