# :lemon::dizzy: **LEMONsc: Learning the Evolution of Massive Objects in Nuclear star clusters**

The formation of massive compact objects remains a central open problem in contemporary astrophysics. Investigating plausible formation pathways relies on extensive computational experiments to assess how differing physical assumptions shape the formation and subsequent growth of intermediate- and supermassive black holes (IMBHs and SMBHs). Here we present an experimental framework of machine learning (ML)–based surrogate models designed to approximate the evolutionary trajectories of the most massive objects in collision dominated systems. Our models are trained on datasets derived from both direct N‑body integrations and MOCCA simulations, covering a broad range of black-hole eficiency runs. By employing representative astrophysical initial conditions, this framework enables systematic exploration of candidate formation and evolutionary scenarios for massive objects.

---
## :computer::open_file_folder: **Project Structure:**
Our repository follows the intended structure:

```
LEMONsc/
    │_ rawdata/                # gitignore. The authors reserve the rights to share the data upon reasonable request.
    │_ datasets/               # gitignore. The authors reserve the rights to share the data upon reasonable request.
    |_ logs/                   # gitignore. Store the output of the functions for debug.
    |_ jobs/                   # main working functions and experiments.
    │_ src/                    # source scripts. Processing, modeling, and additional funtions.
    │_ notebooks/              # interactive jupyter notebooks for test, vizualization and tutorials.
    │_ Dockerfile              # proposed image for running the experiments.
    │_ .dockerignore           # avoid memory issues while creating the container set-up.
    │_ core-build-container.sh # bash script for building the container.
    │_ core-run-container.sh   # bash script for running the container.
    │_ requirements.txt        # python libraries to run the experiments.
    |_ README.md               # main changes and comments.
    |_ .gitignore              # elements to ignore.
```

Further information regarding the different pipelines provided,  the models evaluated or the datasets integrated can be found on their respective `./src/*/README.md` files. We also provide interactive jupyter-notebooks to test the different custom functions.
---

## :question: **Installation:**
 - Clone the repository:
```
git clone https://github.com/Milo-Jsb/LEMONsc.git
cd LEMONSc
```

- Set up your working envirioment:
We strongly suggest the use of Docker to perform your experiments, a docker image is contained in Dockerfile. For a containerized setup, use the provided scripts:
```
bash core-build-container.sh
```
Please have in mind that you need to expose an available port when using a remote host.

---
## :ballot_box_with_check: **How to run:**

You can run the container using:
```
bash core-run-container.sh
```

This script finds an available port to expose and open the container, feel free to update or change the instruction. After running the container from the terminal use the following instruction to run the interactive notebooks:

```
jupyter [notebook|lab] --ip 0.0.0.0 --port 8889 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password=''
```

Or you can recreate our experiments using the scripts provided in the `/jobs/` folder. 

¡Have fun and contact us if you have any suggestions :sunglasses:!

---
