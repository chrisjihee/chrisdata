from chrisbase.data import AppTyper

import chrisdata.wikidata
import chrisdata.wikipedia

main = AppTyper()
main.add_typer(chrisdata.wikidata.app, name="wikidata")
main.add_typer(chrisdata.wikipedia.app, name="wikipedia")


@main.command()
def hello():
    print("Hello Chrisdata!")


@main.command()
def bye(name: str):
    print(f"Bye {name}")


if __name__ == "__main__":
    main()
