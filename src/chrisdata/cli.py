import chrisdata.net.check

from chrisbase.data import AppTyper

main = AppTyper()


@main.command()
def hello():
    print("Hello Chrisdata!")


@main.command()
def bye(name: str):
    print(f"Bye {name}")


if __name__ == "__main__":
    main.add_typer(chrisdata.net.check.app, name="net")
    main()
