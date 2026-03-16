# HAML-UCI-Adult-Classification
This project involves training and deploying a fair classification model on the UCI Adult Dataset as part of the Advanced Data Analysis Methods Laboratory at BME.

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