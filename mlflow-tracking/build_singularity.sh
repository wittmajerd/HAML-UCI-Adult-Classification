#!/bash/bin

echo "Copy def file to tmp"
cp ~/nr_haml2025/HAML-UCI-Adult-Classification/mlflow-tracking/mlflow.def /tmp/mlflow.def
echo "Build singularity image"
singularity build --fakeroot /tmp/mlflow.sif /tmp/mlflow.def
echo "Copy singularity image back to workdir"
cp /tmp/mlflow.sif ~/nr_haml2025/HAML-UCI-Adult-Classification/mlflow-tracking/
echo "Clean up temporary files"
rm /tmp/mlflow.sif
rm /tmp/mlflow.def