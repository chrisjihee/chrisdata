import chrisdata.net
import chrisdata.wikidata
from chrisbase.data import AppTyper

main = AppTyper()
main.add_typer(chrisdata.net.app, name="net")
main.add_typer(chrisdata.wikidata.app, name="wikidata")


@main.command()
def hello():
    print("Hello Chrisdata!")


@main.command()
def bye(name: str):
    print(f"Bye {name}")


if __name__ == "__main__":
    main()
