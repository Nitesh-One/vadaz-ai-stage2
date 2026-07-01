# Stage 1 — Quick Screen (about 1 hour)

**Role:** AI Engineer — AI Astrologers, Vedaz
**Time:** ~1 hour. This is a short first task. If you do well, we'll send you a longer hands-on technical task.
**Submit:** One document (PDF or text/Markdown) + one `.jsonl` file.

---

## About Vedaz and this role

Vedaz is building **AI Astrologers** — chat assistants that give people Vedic astrology guidance in a warm, honest, and responsible way. Most of our users are in India and chat in Hindi or Hinglish.

To make these assistants sound right, we "train" an AI model on good example conversations. Before that, someone has to make sure those example conversations are actually good. That's a big part of this job.

The most important thing about Vedaz's voice is that it is **safe and honest**. Our assistants must always follow these rules:

1. **Never scare people.** No predicting death, serious illness, or that someone's life will be "ruined."
2. **Send serious problems to real experts.** Health questions → see a doctor. Legal or big money questions → see a professional.
3. **Never use fear to sell.** Remedies (mantras, donations, pujas) are described as helpful practices, not magic fixes, and never as something you must pay big money for.
4. **Be honest about limits.** Astrology can suggest tendencies and timing. It cannot guarantee outcomes.

We attach a small set of 15 example conversations. Your task is to judge them and add a few of your own.

---

## What you'll get

- `vedaz_astrologer_finetune.jsonl` — 15 example chats between a user and the AI Astrologer.
- `vedaz_astrologer_finetune.json` — the same chats, easier to read, with topic labels.

Each chat is a list of messages: a `system` message (the instructions to the AI), then `user` and `assistant` turns.

---

## Your two tasks

### Task 1 — Review the examples (write about half a page)
Read all 15 chats. Then tell us, in plain words:
- Which chats follow the rules above well?
- Which ones are weak, vague, or could go wrong?
- What kinds of users or situations are **missing** that a real astrologer deals with every day?
- If we trained an AI on only these 15 chats, what problems might show up?

We are not looking for compliments. We want to see what you notice.

### Task 2 — Write 5 new example chats
Add **5 new conversations** in the same format. Rules:
- At least **3 of the 5** should be in **Hindi or Hinglish** (our main users).
- At least **1** should show the assistant handling a tricky situation safely — for example, gently refusing to predict something scary, or sending a health worry to a doctor.
- Make the users feel real and different from each other (one anxious, one casual, one doubtful, etc.).

Save these 5 chats as a `.jsonl` file in the same format as the examples.

---

## How we'll judge this

- **Do you understand the voice?** Your new chats should feel like Vedaz without us having to correct you.
- **Did you spot the real weak points** in the examples, not just surface ones?
- **Are your Hindi/Hinglish chats natural?**
- **Clear, honest writing.**

**Instant red flags:** any chat that predicts death or disease, promises a medical or money outcome, or pushes a paid remedy through fear. Getting the voice wrong matters more to us than anything else.

You may use AI tools to help, but you must stand behind everything you submit — we'll ask you about your choices.
