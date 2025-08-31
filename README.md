# :lemon::dizzy: **LEMONsc: Learning the Evolution of Massive Objects in Nuclear star clusters**

---
## :computer::open_file_folder: **Project Structure:**
```
LEMONsc/
    │_ rawdata/                # gitignore. The authors reserve the rights to share the data upon reasonable request.
    │_ datasets/               # gitignore. The authors reserve the rights to share the data upon reasonable request.
    |_ logs/                   # gitignore. Store the output of the functions for debug.
    |_ scripts/                # main working functions and experiments.
    │_ src/                    # source scripts. Data preprocessing and additional funtions.
    │_ notebooks/              # interactive jupyter notebooks for test, vizualization and tutorials.
    │_ Dockerfile              # proposed image for running the experiments.
    │_ core-build-container.sh # bash script for building the container.
    │_ core-run-container.sh   # bash script for running the container.
    │_ requirements.txt        # python libraries to run the experiments.
    |_ README.md               # main changes and comments.
    |_ .gitignore              # elements to ignore.
```
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
jupyter notebook --ip 0.0.0.0 --port 8883 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password=''
```

Or you can recreate our experiments using the scripts provided in the `/scritps/` folder (check the respective README folder for more info). 

¡Have fun and contact us if you have any suggestions :sunglasses:!

---
