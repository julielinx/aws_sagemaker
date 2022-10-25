# SageMaker Studio, Production Pipelines, and MLOps

There are a lot of considerations in moving from a local model used to train and predict on batch data to a production model that makes predictions in real time. This series of posts explores how to create an MLOps compliant production pipeline using AWS's SageMaker Studio.

SageMaker Studio is a suite of tools that helps manage the infrastructure and collaboration for a machine learning project in the AWS ecosystem. Some of the biggest advantages of SageMaker Studio include:

- Ability to spin up hardware resources as needed
- Automatically spin down hardware resources once the task is complete
- Ability to create a pipeline to automate the machine learning process from preprocessing data through deploying the model