FROM python:3.8-slim-buster
COPY requirements.txt /opt/app/requirements.txt
COPY . /opt/app
WORKDIR /opt/app
RUN apt-get -y update
RUN apt-get -y install gcc
RUN pip3 install -r requirements.txt
RUN pip3 install pandas==1.3.5 scikit-learn==1.2.2 sagemaker-containers
ENV PYTHONUNBUFFERED=TRUE
WORKDIR /opt/app/ecr_dir/
ENTRYPOINT ["python3"]
