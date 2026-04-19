# HAML-UCI-Adult-Classification
This project involves training and deploying a fair classification model on the UCI Adult Dataset as part of the Advanced Data Analysis Methods Laboratory at BME.

## Start mlflow server on HPC cluster
First you must have set up an ssh key for this to work
#### Conecting via SSH
```
ssh -L 5000:localhost:5000 -i <path_to_ssh_key> nr_hafb@komondor.hpc.kifu.hu
```
* -L specifies an SSH tunnel, so your localhost's 5000 port will be redirected to the HPC cluster's port 5000
* -i specifies the path to the ssh key's file
During connection you will see in the terminal a promt about Two factor auth. For this copy the promted link and log in to EduID in a browser. After successfully login get back to the terminal and hit ENTER, than you should get in to the HPC cluster.
#### Run singualrity
* First go to the directory where singularity files are located and activate singularity module:
```
cd nr_haml2025/HAML-UCI-Adult-Classification/mlflow-tracking
module load singularity
```
* Than you can run the prebuilt singularity container:
```
 singularity run --bind /home/nr_hafb/nr_haml2025/HAML-UCI-Adult-Classification/mlruns:/app/mlruns mlflow.sif
```
Where bind is essential, this is out mount of mlflow files.

#### Tunnel to HPC
If you use the command from run singualrity section an ssh tunnel is already created.
If not you have to use the following command (precondition: mlflow server is already ran by someone and yo just need connection)
```
ssh -N -L 5000:localhost:5000 nr_hafb@vn01
```

### Docker commands
Currently the one container is used for a unified testing environment.
* Build: 
```
docker build -t haml-devenv .
```
* Run:
```
docker run -it --rm --network=host -p 5000:5000 -t haml-devenv /bin/bash
```