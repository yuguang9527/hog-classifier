# docker file to build and run solution.py
#FROM #base image    
#COPY #files to copy
#RUN #install dependencies
#CMD #command to run the training job

# base image
FROM python:3.10-slim

# working directory
WORKDIR /app

# copy requirements.txt
COPY requirements.txt .

# install dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# copy the rest of the files
COPY . .

# run the training job  
CMD ["python", "solution.py"]