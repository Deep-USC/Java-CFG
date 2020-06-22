#!/bin/bash

# parse the source code
python3 extract.py /funcom_processed/functions.json

# run progex json
mkdir -p json
java -jar progex-v3.4.5/progex.jar -outdir ./json -cfg -lang java -format json ./java

# run progex dot
mkdir -p dot
java -jar progex-v3.4.5/progex.jar -outdir ./dot -cfg -lang java -format dot ./java
