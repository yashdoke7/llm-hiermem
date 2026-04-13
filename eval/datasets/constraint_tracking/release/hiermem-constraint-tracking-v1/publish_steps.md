# Publishing Steps

## Hugging Face
1. Create a dataset repository.
2. Upload the `data/` folder, `dataset_manifest.json`, and `dataset_card.md`.
3. Add the paper citation in the dataset card.
4. Verify that downloads preserve filenames and checksums.

## Zenodo
1. Create a new record for the paper version.
2. Upload the same bundle as a release snapshot.
3. Mint the DOI and record it in the paper and README.

## Kaggle
1. Create a dataset mirror from the same files.
2. Add a notebook showing how to load one JSON file.
3. Link back to Hugging Face and Zenodo in the dataset description.
