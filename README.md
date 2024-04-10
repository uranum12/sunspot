# python sunspot code

## install

```sh
pip install -r requirements.txt
```

## develop

```sh
poetry install
poetry run invoke fmt
poetry run invoke lint
poetry run invoke test --cov
```
