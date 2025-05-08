# docker file to build and run solution.py
#FROM #base image    
#COPY #files to copy
#RUN #install dependencies
#CMD #command to run the training job

# base image
FROM python:3.10-slim

# working directory
WORKDIR /app

# working directory
WORKDIR /app

# copy files
COPY . .

# install dependencies
RUN pip install -r requirements.txt

# run the training job  
CMD ["python", "solution.py"]