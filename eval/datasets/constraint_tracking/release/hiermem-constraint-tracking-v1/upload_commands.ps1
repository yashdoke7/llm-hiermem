# PowerShell upload templates

# Hugging Face
# hf auth login
# hf repo create hiermem-constraint-tracking-v1 --type dataset --yes
# hf upload hiermem-constraint-tracking-v1 .\eval\datasets\constraint_tracking\release\hiermem-constraint-tracking-v1\data .
# hf upload hiermem-constraint-tracking-v1 .\eval\datasets\constraint_tracking\release\hiermem-constraint-tracking-v1\dataset_manifest.json .
# hf upload hiermem-constraint-tracking-v1 .\eval\datasets\constraint_tracking\release\hiermem-constraint-tracking-v1\dataset_card.md .

# Zenodo
# Upload the same folder contents through the Zenodo web UI or API client.

# Kaggle
# kaggle datasets init -p .\eval\datasets\constraint_tracking\release\hiermem-constraint-tracking-v1
# kaggle datasets create -p .\eval\datasets\constraint_tracking\release\hiermem-constraint-tracking-v1
