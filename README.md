# Discord EasyOCR BOT (Dockerfile版)

このDockerfile版ならKoyebのHealth Check不要で動作します。

## デプロイ手順
1. このZIPをGitHubにPush
2. Koyebで New Service → GitHubリポジトリを選択
3. RuntimeはDockerモードが自動で選択されます
4. 環境変数 DISCORD_TOKEN を設定
5. Instance: Small(512MB) 以上を推奨
