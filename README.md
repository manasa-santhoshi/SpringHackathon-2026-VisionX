# SpringHackathon-2026-VisionX
Spring Hackathon 2026 project by Team Vision X: turning parking video data into intelligent, real-time insights using AI.

Datasets: [Dragon Lake Parking](https://sites.google.com/berkeley.edu/dlp-dataset) and relative [API](https://github.com/MPC-Berkeley/dlp-dataset), [CHAD](https://github.com/TeCSAR-UNCC/CHAD?tab=readme-ov-file)

## How to run the code

First of all, pull the submodule contents:
```
git submodule update --init
```

Then, create a virtual environment and install the requirements:
```
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

>[!Important]
> You have to download the data from the sources provided and put them in a `data/` folder with the following structure:
>.
>├── processed
>└── raw
>    ├── CHAD
>    └── DLP
>        ├── json
>        └── raw
