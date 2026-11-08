# GPU Transcriber

A Windows app that turns a recording into a written transcript with speaker labels.
You paste a YouTube link or choose an audio file, then press Play.

The app rents a GPU machine on Vast.ai for the length of the job. The machine
does the heavy work and sends the results back. When the job ends, the app destroys the machine
so it stops billing.

You get three files for each job:

- `transcript.txt` - the plain transcript.
- `transcript_timestamped.txt` - the same text with times.
- `transcript.srt` - subtitles.

Speakers are labeled `SPEAKER_00`, `SPEAKER_01` and so on. The app does not guess names.

## One-time setup

Do these steps in order. You only do them once.

1. **Make a Vast.ai account** at cloud.vast.ai and add prepaid credit.
   About $10 is plenty to start.
2. **Turn on two-factor sign-in** on the Vast.ai account. Do this *before* step 3.
   Vast.ai only lets an API key rent machines once two-factor sign-in is on.
   A key made earlier will not work. Make the key after.
3. **Create a Vast.ai API key** with exactly these permissions:
   - Instance: read and write
   - User: read
   - Miscellaneous: read and write
   - Billing: **off**

   Copy the key right away. Vast.ai shows it only once.
4. **Make a Hugging Face account** at huggingface.co (it provides the speaker-labeling models).
   - Create an access token of type **Read** and copy it.
   - While signed in to that same account, open these two pages and accept the terms on each:
     - https://huggingface.co/pyannote/speaker-diarization-3.1
     - https://huggingface.co/pyannote/segmentation-3.0
5. **Install and start the app.**
   - Windows build: the build makes a folder named `dist\gpu-transcriber`
     (see "For developers" to build it). You can copy that whole folder anywhere.
     Start the app by double-clicking `gpu-transcriber.exe` inside it.
     Keep the exe together with the other files in the folder.
   - From source: `python -m app.main`.

   The first start shows a setup window. Paste both keys. The app checks both before it saves
   anything. It checks that the Vast.ai key works and shows your credit. It checks that the
   Hugging Face token works and that the terms are accepted on both pages.
   The keys are stored in Windows Credential Manager. They are never written to a file.

## How a job works

1. Paste a link or choose a file. The app shows a cautious estimate of time and cost.
2. Press Play. The job moves through these stages:
   1. **Preparing** - downloads the audio (YouTube) or converts your file on this PC.
   2. **Starting machine** - rents a machine and waits for it to boot.
      This is the slowest part, because the machine downloads the program image.
   3. **Uploading** - sends the audio to the machine.
   4. **Transcribing** - turns speech into text.
   5. **Labeling speakers (chunk N of M)** - works out who spoke when, a piece at a time.
   6. **Saving results** - downloads the three files and destroys the machine.
3. When the job is done, use **Open transcript**, **Open timestamped**,
   **Open subtitles** or **Show in folder**.
   The files are kept in `%APPDATA%\GpuTranscriber\jobs\<job id>\output`.

**Cost cap.** Each job gets a spending cap, chosen for you: twice the high end of the estimate,
and never less than $1.00. The job shows its cost so far against the cap. If the cost reaches the
cap, the job stops and the machine is destroyed. Press **Raise cap and continue** to set a higher
cap and carry on from where it stopped.

The cap is not exact. The final cost can pass it by a few minutes of rental. Connecting to a new
machine can retry for up to about 5 minutes. Confirming that the machine is destroyed can take
up to about 100 seconds. After a rental whose result is unclear, the app looks for the machine
for up to 3 minutes. The app cannot always stop during these waits.

**If you close the app during a job.** The app asks you to confirm first. With the app closed,
the rented machine keeps running and charging (usually about $0.20-$0.60 an hour) until the job
finishes. After it finishes, the machine waits up to 3 hours for the app to come back and download
the results, and it keeps charging while it waits. The machine's own safety timer shuts it down
once its time limit is reached (at least 3 hours, longer for long recordings). The spending cap
is enforced only while the app is open. Reopen the app to download your results and shut the
machine down.

If the machine is gone when you reopen the app, the job shows as failed
("The machine was lost while the app was closed."). Start the job again from the beginning.
The app only rents a replacement machine by itself when the machine is lost while the app is
open, and then it tries a limited number of times.

**"A machine may still be running".** This warning means the app could not confirm that a machine
was destroyed. The warning shows the machine's id, and the app keeps trying to remove it.
To be sure, sign in at cloud.vast.ai, open Instances, and destroy any machine whose label starts
with `gputx-` by hand.

## Cost and time

One real run, a 3-minute recording on an RTX A5000 rented at $0.23 per hour:

| Step | Time |
|---|---|
| Starting the machine (image download) | about 3 minutes |
| GPU work (transcribe and label) | about 11 seconds |
| Whole job | 3 minutes 35 seconds |
| Cost | about one cent |

The estimate shown before Play is deliberately cautious. It uses constants for a slower GPU
(T4 class). **It is unconfirmed for long recordings. A 5-hour recording has not been
measured yet.** Treat the estimate and the default cap as a guess until a long run has been done.

## What is not verified yet

- The machine's self-destruct safety timer has not been tested on a real machine. The check for
  it (`--verify-self-destroy` in `scripts/live_smoke.py`) has not been run.
- A 5-hour recording has not been run.
- The Hugging Face "terms accepted" check has not been tried with a real account that accepted
  the terms.
- Real YouTube downloads with `yt-dlp.exe` have not been tested live.
- The Windows build was only self-tested (it starts, finds ffmpeg and finds a secure-storage
  backend). It has not been taken through a full job by hand.
- In the Windows build, saving and reading keys in Credential Manager was not tried. Only an
  8-second launch was done.
- The setup window in the Windows build was not looked at or used.
- What happens when a machine is lost while the app is open (the app rents a replacement) has
  not been tried on a real machine.

## Troubleshooting

| What you see | What it means and what to do |
|---|---|
| "Vast.ai refused the API key (HTTP 401)", possibly mentioning Two Factor Authentication | Turn on two-factor sign-in in Vast.ai, then create a **new** key. Keys made before it was on will not work. Paste the new key under "Accounts and keys…" in the app. |
| "refused the API key (HTTP 403)" | The key lacks a permission. Make a new key with the permissions listed in step 3. |
| Your Vast.ai balance is less than the job's spending cap | Add prepaid credit at cloud.vast.ai. |
| "terms are not accepted on: ..." | Open both model pages while signed in to the account that owns the token, accept the terms, and try again. |
| "Hugging Face did not accept this token" | Copy the whole token again. It must be a Read token. |
| "That video is unavailable" | The video is private, removed, or blocked in your country. |
| "That video is a live stream" | Wait until the stream has ended. |
| "YouTube is asking for a sign-in before it will share this video" | The video is age-restricted, members-only or otherwise needs a YouTube sign-in, and the app can't sign in to YouTube. Download the video another way, then choose the audio or video file in the app instead of pasting the link. |
| A link that is not one video | Playlists and channels are not supported. Paste a link to one video. |
| "A machine may still be running" | See the section above. Check cloud.vast.ai and destroy it by hand if needed. |

**Logs** are in `%APPDATA%\GpuTranscriber\logs\gpu_transcriber.log`. Keys are never written
to the log. The helper `yt-dlp.exe` is kept in `%APPDATA%\GpuTranscriber\tools`. The app does
not update it on a schedule. When a YouTube download fails, the app updates `yt-dlp.exe` and tries
once more. It does this at most once every 24 hours.

## For developers

Run from source (Windows, Python 3.10 or newer):

```
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[app,dev,build]"
.venv\Scripts\python.exe -m app.main
.venv\Scripts\python.exe -m app.main --selftest
```

Run the tests:

```
.venv\Scripts\python.exe -m pytest
.venv\Scripts\python.exe -m pytest --cov=app --cov=common --cov=remote --cov-report=term-missing
```

Build the Windows app (one folder, no console window):

```
powershell -ExecutionPolicy Bypass -File scripts\build_exe.ps1
```

This cleans `build/` and `dist/`, runs PyInstaller with `packaging/gpu-transcriber.spec`, runs
`dist\gpu-transcriber\gpu-transcriber.exe --selftest`, and prints the exe path and folder size.
Copy the whole `dist\gpu-transcriber` folder, not only the exe. The build bundles `app/`,
`common/` and the ffmpeg binary from imageio-ffmpeg. It does not bundle `remote/`: the app never
uploads that code, because the machine runs the copy inside the container image.

The container image is built in GitHub Actions. See `docker/README.md`. `IMAGE` in
`app/settings.py` holds the image digest. Update it after every new image build.

Layout: `app/` is the desktop app, `common/` is code shared with the machine, `remote/` runs on the
rented machine, `docker/` builds the image, `scripts/live_smoke.py` is the paid end-to-end check.
