from typer import Typer

app = Typer()


@app.command()
def hello_chrisdata():
    print("Hello Chrisdata!")


@app.command()
def bye_chrisdata(name: str):
    print(f"Bye {name}")
