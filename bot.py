# ここまでのコードは同じ...

@client.event
async def on_message(message):
    if message.author.bot:
        return

    notify_channel = client.get_channel(NOTIFY_CHANNEL_ID) if NOTIFY_CHANNEL_ID else None

    # ✅ 手動デバッグ入力 (!1234-7-12:34:56)
    if message.content.startswith("!"):
        manual_text = message.content.lstrip("!").strip()
        # manual_text 例: 1234-7-12:34:56
        parts = manual_text.split("-")
        if len(parts) == 3:
            server_num, place_num, time_str = parts
            mode = "警備" if server_num == "1281" else "奪取"
            txt = f"{mode} {server_num}-{place_num}-{time_str}"

            # デバッグは時間そのものをそのまま扱う
            try:
                unlock_dt = datetime.strptime(time_str, "%H:%M:%S")
            except:
                unlock_dt = datetime.min  # 開戦済扱い

            # pending_places に追加
            pending_places[txt] = (unlock_dt, txt, server_num)

            # デバッグは通知制限なしで予約
            if txt.startswith("奪取") and unlock_dt != datetime.min and notify_channel:
                asyncio.create_task(schedule_notification(unlock_dt, txt, notify_channel))

            # 通知チャンネルに即スケジュール送信
            if notify_channel:
                await send_schedule_summary(notify_channel)

            await message.channel.send(f"✅ 手動デバッグ登録完了:\n{txt}")
        else:
            await message.channel.send("⚠️ 手動入力は `!サーバー-駐騎場-時刻` の形式で送ってください")
        return

    # ✅ 画像添付時の処理
    if message.attachments:
        processing_msg = await message.channel.send("🔄 画像解析中…")

        parsed_texts_for_reply = []  # チャンネル返信用

        for attachment in message.attachments:
            img_bytes = await attachment.read()
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            img_np = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

            top_img = crop_top_right(img_np)
            center_img = crop_center_area(img_np)

            top_texts = extract_text_from_image(top_img)
            center_texts = extract_text_from_image(center_img)

            parsed_results, no_time_places, _ = parse_multiple_places(center_texts, top_texts)

            # 解析結果処理
            for dt, txt, server in parsed_results:
                key = txt
                pending_places[key] = (dt, txt, server)
                parsed_texts_for_reply.append(txt)

                # 1281（警備）は通知なし、奪取のみ予約
                if txt.startswith("奪取") and notify_channel:
                    asyncio.create_task(schedule_notification(dt, txt, notify_channel))

            # 開戦済も追加
            for txt in no_time_places:
                pending_places[txt] = (datetime.min, txt, "")
                parsed_texts_for_reply.append(txt)

        # ✅ チャンネル返信は結果だけ
        if parsed_texts_for_reply:
            await processing_msg.edit(content="\n".join(parsed_texts_for_reply))
        else:
            await processing_msg.edit(content="⚠️ 必要な情報が読み取れませんでした")

        # ✅ 駐騎場1〜12揃ったら通知チャンネルにまとめて送る
        places_found = {txt.split("-")[1] for _, txt, _ in pending_places.values() if "-" in txt}
        if len(places_found) >= 12 and notify_channel:
            await send_schedule_summary(notify_channel)