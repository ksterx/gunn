# gunn ドキュメント

`gunn` はイベント駆動のマルチエージェント・オーケストレーションコアです。このサイトではプロダクトの全体像、技術的なディテール、統合パターン、そして API 参照への入り口をまとめています。

## スタートガイド

- リポジトリのセットアップは `README.md` と `CONTRIBUTING.md` を参照してください。
- ドキュメントの構成は下記のナビゲーション、またはページ右側の目次から辿れます。

## ビルドとデプロイ

GitHub Pages に公開する場合はルートディレクトリから次のコマンドを実行し、`docs/_build/html` を Pages の公開対象に設定してください。

```bash
uv sync --group docs
uv run sphinx-build -b html docs docs/_build/html
```

```{toctree}
:caption: ガイド
:maxdepth: 2
:hidden:

product
structure
tech
integration-patterns
web_adapter
errors
build-and-deploy
api/index
```
