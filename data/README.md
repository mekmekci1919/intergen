# Data directory

Place the two input Excel files here using these exact default filenames:

- `collision_2023_clevand.xlsx`
- `day_bike_share.xlsx`

The experiment script reads them from `./data/` by default.

If your datasets are stored elsewhere, either copy `.env.example` to `.env` and set the paths there, or export one of these environment variables:

- `INTERGEN_DATA_DIR`
- `INTERGEN_COLLISION_FILE`
- `INTERGEN_BIKE_FILE`

Dataset files are ignored by Git by default to avoid accidentally publishing data that may have separate licensing or redistribution restrictions.
