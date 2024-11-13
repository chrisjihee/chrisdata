import chrisdata.ner
import chrisdata.net
import chrisdata.wikidata
import chrisdata.wikipedia
from chrisbase.data import AppTyper

main = AppTyper()
main.add_typer(chrisdata.net.app, name="net")
main.add_typer(chrisdata.ner.app, name="ner")
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
