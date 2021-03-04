import pandas as pd
import argparse
import shutil
import os

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--metadata", required=True)
ap.add_argument("-o", "--output", required=True)
args = vars(ap.parse_args())

csvPath = os.path.sep.join([args["metadata"], "metadata.csv"])
df = pd.read_csv(csvPath)

for (i, row) in df.iterrows():
	if row["finding"] != "COVID-19" or row["view"] != "PA":
		continue

	imagePath = os.path.sep.join([args["metadata"], "images",
		row["filename"]])

	if not os.path.exists(imagePath):
		continue

	filename = row["filename"].split(os.path.sep)[-1]
	outputPath = os.path.sep.join([args["output"], filename])

	shutil.copy2(imagePath, outputPath)