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
Currently with the mlflow on HPC setup the procedure is the following:
* Build (assuming you are in the root of the project): 
```
docker build -t haml-devenv .
```
* Run:
```
docker run -it --rm -p 5000:5000 -t haml-devenv /bin/bash
```
### Tracking during training
You can check ml-flow-test python script under mlflow-tracking to see an example how the tracking is done
The setting of tracking URI is importatnt, you must select one based on you run the training on local machine or indise the docker:
```
    mlflow.set_tracking_uri("http://host.docker.internal:5000") # <-- This line is if we are using the dev container
    mlflow.set_tracking_uri("http://localhost:5000") # <-- This line is if we are running the script locally
    mlflow.set_experiment("Default")
```

## Fairness (Méltányosság) Metrikák

A kódban szereplő négy fairness metrika a [Fairlearn](https://fairlearn.org/v0.10/user_guide/assessment/common_fairness_metrics.html) könyvtárból származik. Alapvetően két fő koncepciót mérnek (Demographic Parity és Equalized Odds), mindkettőt kétféleképpen: különbségként (difference) és arányként (ratio).

### 1. Demographic Parity (Statisztikai Paritás)
Ez azt vizsgálja, hogy a modell **milyen arányban ad pozitív jóslatot (pl. >50K jövedelem) az egyes védett csoportoknak** (pl. férfiak vs. nők). Azt várja el, hogy a pozitív jóslatok aránya független legyen a védett tulajdonságtól.

*   **`demographic_parity` (Difference - Különbség):**
    *   **Mit jelent?** A legnagyobb és a legkisebb pozitív jóslati arány (Positive Rate) közötti különbség a különböző csoportok között. (Például: ha a férfiak 30%-a kap pozitív jóslatot, a nőknek pedig 10%-a, akkor a különbség 0.2).
    *   **Mi a jó érték?** A tökéletes érték a **0.0**. Minél közelebb van a nullához, annál "fairebb" a modell ezen metrika szerint.

*   **`demographic_parity_ratio` (Arány):**
    *   **Mit jelent?** A legkisebb pozitív jóslati arány osztva a legnagyobb pozitív jóslati aránnyal. (Az előző példával: 10% / 30% = 0.33).
    *   **Mi a jó érték?** A tökéletes érték az **1.0**. A gyakorlatban, a *Four-Fifths Rule* (Négyötödös szabály) alapján a **0.8 feletti** értékeket (0.8 és 1.0 között) gyakran már elfogadhatónak tekintik az iparban.

### 2. Equalized Odds (Kiegyenlített Esélyek)
Ez egy szigorúbb metrika. Nem csak a pozitív jóslatok arányát nézi, hanem a modell **hibázásait** is. Azt várja el, hogy a modell ugyanakkora arányban találja meg a *ténylegesen* pozitív eseteket (True Positive Rate - TPR), és ugyanakkora arányban adjon téves riasztást a *ténylegesen* negatív esetekre (False Positive Rate - FPR) minden csoportban.

*   **`equalized_odds` (Difference - Különbség):**
    *   **Mit jelent?** Kiszámolja a TPR és FPR különbségeit a csoportok között, és a kettő közül a **legnagyobb eltérést** adja vissza.
    *   **Mi a jó érték?** A tökéletes érték a **0.0**. Minél kisebb (közelebb van nullához), annál jobb, vagyis a modell nem hibázik szignifikánsan többet/másképp az egyik csoport kárára a másikhoz képest.

*   **`equalized_odds_ratio` (Arány):**
    *   **Mit jelent?** Hasonló a különbséghez, de hányadosként számolja ki a TPR-ek és FPR-ek arányát a csoportok között, és a legkisebb (legrosszabb) arányt adja vissza.
    *   **Mi a jó érték?** A tökéletes érték az **1.0**. Minél közelebb van az 1-hez, annál jobb. Ahogy a másik rátánál, itt is a bűvös határ sokszor a 0.8.

### Összefoglalva
*   **Ha "Difference" (különbség)** a metrika neve: A cél a **0.0** elérése. (Pici eltérés, pl. 0.05 még jó lehet, egy 0.2..0.3 már komoly torzítást jelez).
*   **Ha "Ratio" (arány)** a metrika neve: A cél az **1.0** elérése. (Ideális esetben > 0.8).