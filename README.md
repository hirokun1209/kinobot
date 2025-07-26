# Discord EasyOCR BOT (軽量版Docker)

- ベース: python:3.10-bullseye
- torchは2.0.1+cpuで軽量化
- FlaskでヘルスチェックHTTPサーバー起動するのでKoyebがUnhealthyにならない

## デプロイ手順
1. このZIPをGitHubにPush
2. Koyeb → New Service → GitHubリポジトリを選択
3. Dockerモードが自動選択
4. 環境変数 DISCORD_TOKEN を設定
5. Instance: Small(512MB)以上推奨
