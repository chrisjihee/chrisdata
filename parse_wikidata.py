import logging
import gzip

from chrisbase.data import AppTyper

logger = logging.getLogger(__name__)
app = AppTyper()


@app.command()
def parse():
    num_line = 100
    with gzip.open("/fed/Wikidata/latest-truthy-nt-gz/latest-truthy.nt.gz") as f:
        n = 0
        for line in f:
            print(line)
            n += 1
            if n > num_line:
                break
        print(f"f={f}")


if __name__ == "__main__":
    app()
