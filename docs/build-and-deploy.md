# ドキュメントビルドと公開手順

## 依存関係の準備

ドキュメント関連の依存を `docs` グループで管理しています。初回は次を実行してインストールしてください。

```bash
uv sync --group docs
```

## ローカルビルド

HTML を生成するにはプロジェクトルートで以下を実行します。

```bash
uv run sphinx-build -b html docs docs/_build/html
```

Makefile のショートカットも利用できます。

```bash
cd docs
make html
```

ビルド結果は `docs/_build/html` に出力されるため、ブラウザで `index.html` を開いて確認できます。

## リンク検証

外部リンクを検証するには `linkcheck` ビルダーを利用します。

```bash
cd docs
make linkcheck
```

## GitHub Pages へのデプロイ

1. GitHub リポジトリ設定の「Pages」でソースを `main` ブランチの `/docs/_build/html` に指定します。
2. CI から公開する場合は、ジョブ内で `uv sync --group docs` → `uv run sphinx-build -b html docs docs/_build/html` を実行し、成果物をコミットまたは Pages アーティファクトとしてアップロードします。
3. Pages は `.nojekyll` を自動生成するため、追加設定は不要です。テーマやメタ情報は `docs/conf.py` で調整できます。

## API リファレンスを更新する際のヒント

- `sphinx-apidoc -o docs/api src/gunn` を利用すると自動でスタブを生成できます。
- 生成済みファイルは必要に応じて MyST Markdown に書き換え、`api/index.md` の toctree に追加してください。
- 新しいモジュールを公開する場合は docstring とタイプヒントを整備するとドキュメントが読みやすくなります。
