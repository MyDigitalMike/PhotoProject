# Meme Collection Guide

## Add a new reaction

1. Create a folder under `assets/memes` using the meme key.
   Example: `assets/memes/absolute_cinema`.

2. Add image files with one of these extensions:
   `.jpg`, `.jpeg`, `.png`, `.webp`, `.gif`.

3. Add or tune a profile in `assets/memes/catalog.json`.

Without API providers, the matcher only activates profiles whose key has real
image files. With API providers enabled, profiles can also be activated by
remote results.

## Useful search query patterns

Use image search for discovery, then save only images you have permission to
use.

```text
absolute cinema meme filetype:jpg
absolute cinema reaction meme filetype:png
wtf reaction meme filetype:jpg
what bro reaction meme filetype:png
confused reaction meme filetype:jpg
side eye reaction meme filetype:png
chef kiss meme filetype:jpg
mind blown reaction meme filetype:png
```

For automation, prefer an official image-search API or a curated dataset/source
with clear licensing. Avoid direct Google Images scraping inside the app.

## API-backed memes

The app can now use local images plus external providers.

Configuration lives in:

```text
assets/memes/api_providers.json
```

Current providers:

```text
Imgflip: enabled by default, no key required for top meme templates.
GIPHY: enabled when GIPHY_API_KEY exists in your environment.
```

Useful PowerShell setup:

```powershell
$env:GIPHY_API_KEY="your_giphy_key"
```

You can also create a local `.env` file:

```text
GIPHY_API_KEY=your_giphy_key
```

`.env.example` is only a template and is not meant to hold real secrets.

On startup, the console prints masked provider status:

```text
Meme API provider: giphy enabled=True, env=GIPHY_API_KEY, key=abcd...wxyz
```

If later logs show:

```text
providers=['imgflip']
```

then GIPHY is not active in that Python process. The usual cause is putting the
key in `.env.example` instead of `.env`, or setting `$env:GIPHY_API_KEY` in a
different terminal.

When a remote search happens, you should see lines like:

```text
API repository search: key=wtf_bro providers=['giphy', 'imgflip'] ...
API search: provider=giphy key=wtf_bro query='wtf reaction meme'
API search result: provider=giphy key=wtf_bro candidates=6
```

GIPHY often returns `.gif` URLs. OpenCV handles the final window, but Pillow is
used to decode animated GIF/WebP frames. The repository returns the correct frame
on each app loop, so remote and local GIFs can animate. If a remote candidate
still cannot be decoded, the API repository skips it and tries another candidate
before falling back to local assets.

To avoid spending API calls too quickly, the remote repository uses three pacing
settings:

```json
"minimum_remote_display_seconds": 5,
"remote_request_cooldown_seconds": 8,
"media_cache_seconds": 900
```

`minimum_remote_display_seconds` keeps the current remote meme visible long
enough to actually see it.

`remote_request_cooldown_seconds` prevents new provider searches while the last
remote search is still fresh.

`media_cache_seconds` reuses already downloaded and decoded remote images/GIFs
for the same meme key before calling providers again.

Set `"prefer_remote": true` in `api_providers.json` when you want APIs to win
over local folders. The safer default is `false`: local first, API fallback.

Profiles can provide search terms:

```json
"search_terms": [
  "absolute cinema meme",
  "cinema reaction meme"
]
```

If `search_terms` is missing, the app creates a fallback query from the profile
key, for example `absolute cinema reaction meme`.

## API-backed emotions

DeepFace remains the default local analyzer. Hugging Face can be used as an
optional fallback when DeepFace returns low confidence.

PowerShell setup:

```powershell
$env:HUGGINGFACE_API_TOKEN="your_hf_token"
$env:HF_EMOTION_MODEL="your/image-classification-model"
```

The Hugging Face model should return labels that can be mapped to:
`happy`, `sad`, `angry`, `surprise`, `fear`, `disgust`, or `neutral`.

## Profile fields

`priority` gives a profile a base score.

`min_total_score` is the minimum score required before the profile can win.

`emotion_weights` multiplies DeepFace emotion percentages.

`signal_weights` adds score when a visual signal is active.

`required_signals` must all be active.

`any_signals` requires at least one listed signal.

`blocked_signals` disqualifies the profile when any listed signal is active.

`min_scores` requires each listed emotion to reach its threshold.

`any_min_scores` requires at least one listed emotion to reach its threshold.

## Visual signals

Supported signal names include:

```text
mouth_open
mouth_closed
mouth_wide_open
mouth_smile
mouth_puckered
hand_near_mouth
hand_near_forehead
hand_near_chin
hand_near_temple
hand_near_face
hands_near_cheeks
no_hands
any_hands
one_hand
two_hands
thumbs_up
open_palm
one_open_palm
two_open_palms
```

Useful aliases:

```text
close_mouth
closed_mouth
mouth_close
smile
smiling
kiss_lips
puckered_lips
duck_face
kiss_mouth
wide_open_mouth
big_mouth_open
hands_visible
hand_visible
any_hand
hand_near_head
hand_face
no_hand
single_hand
both_hands
single_open_palm
```

If a profile uses a signal that is not supported, it will never match because
unknown signals evaluate as `False`.
