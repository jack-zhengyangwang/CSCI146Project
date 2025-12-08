import kagglehub
import pandas as pd
# Download latest version
path = kagglehub.dataset_download("visualize25/valorant-pro-matches-full-data")

print("Path to dataset files:", path)

