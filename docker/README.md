# Container image

The image the rented Vast.ai machine runs (`docker/Dockerfile`). It holds `/app/common`, `/app/remote`,
ffmpeg, sshd, the `vastai` CLI and the pinned Python packages, plus the faster-whisper
`large-v3-turbo` weights. It contains no secrets and no pyannote models (those are gated and are
downloaded at job start with the user's Hugging Face token). This PC has no Docker, so the image is
built in GitHub Actions.

## Build it

1. Push the repo to GitHub (private repo `gpu-transcriber`).
2. Actions tab, workflow "Build image", "Run workflow" (or push a tag such as `image-1`).
3. The workflow builds the image, runs `tests/test_image_contract.py` against it, and pushes to
   `ghcr.io/<owner>/gpu-transcriber` only if the test passes. A failing test means nothing is pushed.

## Read the digest

Open the finished run, then the job summary. It shows `ghcr.io/<owner>/gpu-transcriber@sha256:...`
and the image size. That digest reference is the value to use; tags can move, digests cannot.

## Make the package public

Vast.ai pulls the image without credentials, so the package must be public. This is safe because the
image contains no secrets and no gated models.

GitHub profile, Packages, `gpu-transcriber`, Package settings, Danger Zone, "Change visibility",
Public. (Optionally link the package to the repo on the same page.)

## Freeze the pins after the first green build

`docker/requirements.txt` pins only what is known; the rest is loose and marked
`TODO(pin-after-first-build)`.

1. From the green run, download the `pip-freeze` artifact (`pip-freeze.txt`).
2. Copy the versions of the packages listed in `requirements.txt` (and their dependencies if you want
   full reproducibility) from it into `docker/requirements.txt`. Do not add torch, torchaudio or
   torchvision; they come from the base image. `docker/constraints.txt` pins torch and torchaudio
   to the base image's 2.5.1 so pip cannot upgrade them (the first build failed that way); change it
   together with the base image tag, never alone.
3. Commit, run the workflow again, and use the new digest.

## Use the digest in the app

Set `IMAGE` in `app/settings.py` to the digest reference from the job summary, for example
`IMAGE = "ghcr.io/<owner>/gpu-transcriber@sha256:<digest>"`. It is currently the placeholder
`gpu-transcriber:unpublished`.

## Run the contract test by hand

With Docker installed:

```
GPU_TRANSCRIBER_TEST_IMAGE=<image> python -m pytest tests/test_image_contract.py -m integration
```

Without Docker or without the variable, the tests are skipped.
