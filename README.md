# BabyVLM dataset downloader

Download data from Databrary for use with BabyVLM-V2 playground. Please make sure you have a Databrary account and can normally access [BabyVLM-V2](https://databrary.org/volume/1981), [SAYCam](https://databrary.org/volume/564), and [BabyView](https://databrary.org/volume/1882) volumes on a web browser, before proceeding.

## Table of contents

- [Setup](#setup)
- [Download filtered pretraining data](#download-filtered-pretraining-data)
  - [Extract the downloaded frames](#extract-the-downloaded-frames)
  - [Using the data for training](#using-the-data-for-training)
- [Download raw data](#download-raw-data)
  - [Step 1 — Utterance JSONs](#download-raw-data)
  - [Step 2 — SAYCam videos](#download-raw-data)
  - [Step 3 — BabyView videos](#download-raw-data)
- [Download DevCV toolbox data](#download-devcv-toolbox-data)

## Setup

Requires Python 3.9+.

```bash
conda create -n databrary python=3.11
conda activate databrary
pip install -r requirements.txt
playwright install chromium
```

On first run the scripts open a headless browser and prompt for your Databrary
email and password. If your account has two-factor authentication enabled you
will also be prompted for the code once. After that the session is saved to
`~/.databrary-browser-profile/` and subsequent runs require no interaction.

**Optional:** to skip the prompts, save your credentials in a `.env` file next
to the scripts (never commit it):

```
DATABRARY_USERNAME=your@email.com
DATABRARY_PASSWORD=yourpassword
```
<!-- 
```bash
chmod 600 .env
``` -->

---

## Download filtered pretraining data

`filtered_pretrain_download.py` downloads the filtered SAYCam and BabyView pretraining data
(MJPEG files, frame indexes, and training JSONs) from the
[BabyVLM Databrary volume](https://databrary.org/volume/1981).
Total download size is about 560 GB; allow additional space for extracted frames.

```bash
python filtered_pretrain_download.py get 1981 --out /path/to/storage
```

This downloads all files in `SAYCam_pretrain`, `BabyView_pretrain`,
and `merged_pretrain`.
Progress is saved continuously — if it stops for any reason, re-run the same
command to resume.

For long runs, use `tmux` or `nohup` so it survives logout:

```bash
nohup python filtered_pretrain_download.py get 1981 --out /path/to/storage > download.log 2>&1 &
tail -f download.log
```

### Extract the downloaded frames

Run extraction once for SAYCam and once for BabyView. Each source directory must
contain `images.index.json` and its MJPEG file(s): one for SAYCam, all ten chunks
for BabyView. Both `.mjpeg` and Databrary's `.mjpg` filenames are accepted.
Training JSONs such as `single_image.json` are not extraction inputs.

```bash
python filtered_pretrain_download.py extract \
  /path/to/storage/78177_SAYCam_pretrain \
  --out /path/to/frames/78177_SAYCam_pretrain \
  --workers 4

python filtered_pretrain_download.py extract \
  /path/to/storage/78187_BabyView_pretrain \
  --out /path/to/frames/78187_BabyView_pretrain \
  --workers 4
```

Replace the source paths if you stored the MJPEG files elsewhere. Keep the output
directory names shown above so the training JSONs can find both datasets under
one image root. The indexes contain paths starting with `images/`, so the result is:

```text
/path/to/frames/
├── 78177_SAYCam_pretrain/
│   └── images/*.jpg
└── 78187_BabyView_pretrain/
    └── images/*.jpg
```

<!-- The script copies each JPEG's indexed byte range without decoding or re-encoding
it, and verifies its SHA-256 when provided by the index. No login, network access,
or FFmpeg is needed. The MJPEG files and indexes are retained unchanged.
Re-run the same commands after an interruption: valid existing frames are checked
and skipped; invalid frames cause an error rather than being overwritten. -->

The output directory must be separate from the source: it cannot be the source,
inside it, or an ancestor of it. Allow roughly another **557 GB** plus filesystem
overhead for **2.39 million JPEGs**, enough available inodes, and several GB of RAM
for loading the indexes. Use `tmux` for long extractions. `--workers` defaults to 4.
The `merged_pretrain` folder contains only training JSONs and needs no extraction.

### Using the data for training

The training scripts are in the [BabyLLaVA-V2](https://github.com/ShawnKing98/BabyLLaVA-V2) repo.
Clone that repo, then edit its training scripts to point at this data.

`--image_folder` is the shared extraction root, `/path/to/frames`, not the
download directory or either dataset's individual `images/` directory.
The training JSONs remain under `/path/to/storage` and reference image paths
relative to this shared root.

**Phase 1** uses `single_image.json`:

```bash
--image_folder /path/to/frames \
--data_path /path/to/storage/78233_merged_pretrain/single_image.json
```

**Phase 2** uses `mix_format.json`:

```bash
--image_folder /path/to/frames \
--data_path /path/to/storage/78233_merged_pretrain/mix_format.json
```

Per-dataset JSONs are also downloaded into `78177_SAYCam_pretrain/` and
`78187_BabyView_pretrain/` (each with `single_image`, `multi_image`, `interleaved`,
and `mix_format`) for experimentation with individual data sources.

---

## Download raw data

`unfiltered_pretrain_download.py` downloads the raw utterance JSONs and unfiltered
video files for both SAYCam and BabyView.

**Step 1 — utterance JSONs** (volume 1981, folder 78240, ~339 MB):

```bash
python unfiltered_pretrain_download.py get 1981 78240 --out /path/to/storage
```

Downloads `saycam_utterances` and `babyview_utterances` into
`/path/to/storage/78240_raw_utterances/`.

**Step 2 — SAYCam videos** (volume 564, 463 sessions, ~450 GB):

```bash
python unfiltered_pretrain_download.py get 564 --sessions --flat --out /path/to/saycam_videos
```

The `--flat` flag saves all files directly into `/path/to/saycam_videos/` with no
subdirectory, so each file is `/path/to/saycam_videos/{video_id}.mp4`. This lets
`saycam_utterances.json` index directly into the downloaded files:

```python
video_path = Path("/path/to/saycam_videos") / (record["video_id"] + ".mp4")
```

Progress is saved continuously — re-run the same command to resume if interrupted.

<!-- ### Download BabyView videos -->

**Step 3 — BabyView videos** (volume 1882, 51 sessions, 16,303 files, ~342 GB):

```bash
python unfiltered_pretrain_download.py get 1882 --sessions --flat --out /path/to/babyview_videos
```

With `--flat`, each file lands at `/path/to/babyview_videos/{rec_id}.mp4` and can
be looked up from `babyview_utterances.json` via the `video_id` field.

---

## Download DevCV toolbox data

The DevCV toolbox evaluation data is in volume 1981 as six finetune folders.
[Evaluate on this data](https://github.com/ShawnKing98/DevCV-Toolbox)

<!-- | Folder ID | Name |
|---|---|
| 78234 | SAYCam_finetune_train |
| 78235 | SAYCam_finetune_val |
| 78236 | SAYCam_finetune_test |
| 78237 | BabyView_finetune_train |
| 78238 | BabyView_finetune_val |
| 78239 | BabyView_finetune_test | -->

```bash
python devcv_download.py --out /path/to/storage
```

Each folder is extracted into a `{folder_id}_{folder_name}/` subdirectory under `--out`.
After downloading, the script automatically creates `merged_devcv/` under `--out` with one
JSON per task×split (e.g. `compare_real_train.json`) combining SAYCam and BabyView entries,
with image paths prefixed by their source folder so they resolve relative to `--out`.
