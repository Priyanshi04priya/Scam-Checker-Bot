# Demo script

Demo media (photos, voice notes, video) is not committed: the team supplies it under
`eval/data/real/media/`. Rehearse once with the real keys before presenting.

## Preparation checklist

- Keys in `.env`: Telegram bot token and webhook secret, Anthropic, Google (fact-check search,
  Safe Browsing, image search), Sightengine. Optional: WhatsApp credentials and an allowlist.
- Services running: database, Redis, the checker tool server, the API and the worker
  (`docker compose up`). Run the migrations.
- Fact-check index filled: run the backfill command so Hindi and English debunks are searchable.
- Sample files in `eval/data/real/media/`: an old photo with a recent false caption, a possibly
  AI-generated image, a scam voice note (and a harmless one), and a short video.
- Bot reachable from the demo phone (webhook set, or polling in a dev setup).

## Expected timings

Text answers in about 5 to 15 seconds, images about 10 to 30 seconds, audio and video up to
60 seconds. A repeated item is answered from the cache in under 3 seconds. If a reply has not
arrived after 90 seconds the bot apologises and asks the user to resend.

## Scenarios

1. **Old photo, new caption.** Forward the old photo with a recent false caption. Expect a
   "Real but old or out of context" reply with the original source link. Point at the source
   link and the explanation of what is wrong with the caption.
2. **Scam text.** Send an English "your account will be blocked, update KYC" message, then a
   Hindi one. Expect "Likely scam" with the reasons (urgency, link, request for an OTP) in the
   same language. Point at the advice not to click or share an OTP.
3. **Debunked claim.** Send a claim that a published fact-check has debunked, then the Hindi
   version. Expect "False/Misleading" with the fact-check link. Point at the link.
4. **Suspicious voice note.** Send a voice note asking for urgent money and an OTP. Expect the
   scam script to be flagged and, if the voice looks artificial, a clearly low-confidence note
   about the voice. Point at the wording: it is a signal, not a verdict.
5. **Possibly AI-generated image.** Send the AI-generated picture. Expect a "Possibly
   AI-generated" label with a medium band and a caution that this is not certain.
6. **Unverifiable claim.** Send a rumour with no source. Expect "Could not verify" and advice to
   check an official source; the bot must not call it genuine. Point at the wording.
7. **Vendor outage.** Unset one key (for example the image-search key), restart the services and
   send the old photo again. Expect a reply that names the check that could not run. Restore
   the key afterwards.
8. **Duplicate.** Re-send a re-compressed copy of an already checked image. Expect an instant
   answer marked as coming from an earlier check.
9. **Two channels.** Send the same item through Telegram and through WhatsApp. Expect the same
   verdict. A WhatsApp number that is not on the allowlist gets no reply.

## If a vendor is down

Say so, show the reply that names the missing check (scenario 7), and continue with the
scenarios that do not depend on that vendor: scam text and unverifiable claims need only the
language model; the cached duplicate needs nothing.
