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

リポジトリには `Publish Docs` ワークフロー（`.github/workflows/docs-pages.yml`）を追加しています。`main` ブランチへの push ごとに Sphinx をビルドし、生成物を GitHub Pages（`gh-pages` ブランチ）へ反映します。

1. 初回のみ GitHub の **Settings → Pages** でソースを「GitHub Actions」に変更してください。
2. ワークフローでは `uv sync --group docs` → `uv run sphinx-apidoc` → `uv run sphinx-build -b html` を順に実行し、`docs/_build/html` をアーティファクト化して Pages にデプロイします。
3. Pages には `.nojekyll` を自動生成してアップロードするため、静的アセット（`_static` など）も確実に配信されます。追加設定は不要です。

手動で公開フローを検証したい場合は、同じコマンド列をローカルで実行し、`docs/_build/html` の内容をそのまま配信対象にすれば GitHub Pages と一致します。

## API リファレンスを更新する際のヒント

- `sphinx-apidoc -o docs/api src/gunn` を利用すると自動でスタブを生成できます。
- 生成済みファイルは必要に応じて MyST Markdown に書き換え、`api/index.md` の toctree に追加してください。
- 新しいモジュールを公開する場合は docstring とタイプヒントを整備するとドキュメントが読みやすくなります。
