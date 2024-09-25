import chrisdata.net.check
from chrisbase.data import AppTyper
from chrisdata.wikidata import wikidata_app

main = AppTyper()
main.add_typer(chrisdata.net.check.app, name="net")
main.add_typer(wikidata_app, name="wikidata")


@main.command()
def hello():
    print("Hello Chrisdata!")


@main.command()
def bye(name: str):
    print(f"Bye {name}")


if __name__ == "__main__":
    main()
