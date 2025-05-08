# Tier2 Take Home Test: PyTorch Classifier

There's no trick question; this take home test is for evaluating your technical abilities.

1. Finish implementing `solution.py` (test routine):
  * If implemented correctly, the following test accuracy should be achieved with the provided weights:
	  * Configuration: Histogram of Oriented Gradients (HOG) — Test accuracy: `56.59%`
  * Run `python solution.py --mode test --feature_type hog --num_unit 64` to test your solution.
   	- You can run `python solution.py --help` to see all available arguments.
2. Train the model with two different learning rates on at least two different architectures (e.g. different numbers of neurons & layers). In total, report a minimum of four train/test results.
3. Complete dockerfile to run `python solution.py --mode train --feature_type hog --num_unit 64` and run this both locally and remotely in the tooling of your choice. 
4. Write at least one paragraph discussing your findings and provide instructions on how to pull and run your Docker image from [Docker Hub](https://hub.docker.com/)


**Environment setup**: `requirements.txt` file is recommended to be used with pip to setup your environment.
```bash
pip install -r requirements.txt
```
We recommend using a virtual environment manager, like `conda` or `venv` - we recommend [pyenv-virtualenv](https://github.com/pyenv/pyenv-virtualenv)


## Submission instructions

Please submit 
1. The executable version of the codebase (with relevant comments).
2. Instructions on how to run the Docker image locally and the steps you used to run training remotely using the same Docker image and the tooling of your choice
3. A written summary as specified in point 4 above.
