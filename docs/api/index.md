# API リファレンス

`gunn` の公開 API を Sphinx の autodoc で自動生成するためのエントリポイントです。必要に応じてサブモジュールごとのページを追加してください。

```{toctree}
:maxdepth: 2
:hidden:

orchestrator
facades
adapters
gunn
```

## 自動生成のワークフロー

1. プロジェクトルートで `sphinx-apidoc -o docs/api src/gunn` を実行します。
2. 生成された `*.rst`/`*.md` を MyST 形式に整えて上記の toctree に追記します。
3. `make html` を実行して、イントロページやサンプルコードが期待通りにレンダリングされるか確認してください。

> **Note**: 自動生成を実施するまでは toctree のプレースホルダ（`orchestrator` など）は空のままでも構いません。不要であれば削除できます。
