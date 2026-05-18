#!/bin/bash

echo "Copy def file to tmp"
cp ~/nr_haml2025/HAML-UCI-Adult-Classification/api/app.def /tmp/app.def
echo "Build singularity image"
singularity build --fakeroot /tmp/app.sif /tmp/app.def
echo "Copy singularity image back to workdir"
cp /tmp/app.sif ~/nr_haml2025/HAML-UCI-Adult-Classification/api/
echo "Clean up temporary files"
rm /tmp/app.sif
rm /tmp/app.def