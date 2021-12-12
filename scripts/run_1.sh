#!/bin/bash
export JARVIS_PATH_CONFIGS=/home/mramados/.jarvis
python /home/mramados/dsn-project/model/preprocess_crossencoder.py 1 > /home/mramados/dsn-project/crossencoder1.log 2>&1
