# Purpose

The purpose of this series of notebooks is to take data from a raw state through preprocessing to model training then deploy an inference pipeline consisting of both the preprocessing estimator and the ML model.

Training data is in `parquet` and `csv` file formats. Inference/prediction data will be passed to the pipeline in `json` format with prediction results being returned in `json` file format.

The pipeline takes about 45 minutes to run on the default `ml.m5.xlarge` instance size.

# Prerequisites and Requirements

## Packages

There are currently two major package version requirements:

- scikit-learn version 1.2-2
- pandas version 1.3.5

As such, notebooks should be run on Data Science 2.0 or Data Science 3.0 images.

## ECR Image

This pipeline demonstrates the use of custom preprocessing, including user defined `classes` passed into a `sklearn` pipeline. To use these, an ECR image is required for preprocessing. The following files have been provided for the purpose of creating an ECR image:

- `Dockerfile`
- `requirements.txt`
- `ecr_dir/transformers.py`

The `image_name` and `dept` variables will need to be updated in the `0b_define_pipe.ipynb` notebook to reflect the file path to your deployed ECR image.

## Kernel

With the default sample size of 10,000, a `ml.t3.medium` notebook kernel should be sufficient for all processes.

# Notebook Contents

1. `0a_write_scripts.ipynb`: this notebook has all the most up to date scripts that are needed to run the pipeline. No changes are needed to run this notebook. Defined scripts include:
    - `create_feats.py`
    - `create_gt.py`
    - `preprocessor_source_dir/transformers.py`
    - `Dockerfile`
    - `requirements.txt`
    - `ecr_dir/transformers.py`
    - `processor_script.py`
    - `preprocessor_source_dir/requirements.txt`
    - `preprocessor_source_dir/processor_model.py`
    - `evaluate.py`
2. `0b_define_pipe.ipynb`: this notebook defines the Sagemaker Pipeline, including all steps, to the point of registering the model. Several variables need to be updated in this notebook to reflect your setup. These include:
    - `image_name`: the image name of your deployed ECR instance
    - `dept`: the department/area/service where the ECR is deployed
    - `tags`: any required or optional tags you want resources to have
3. `2_pipe_prediction_test.ipynb`: this notebook deploys the trained model to an endpoint and sends a prediction. Variables that need to be updated for your particular case include:
    - `tags`: any required or optional tags you want resources to have
    - `version`: multiple runs of the pipeline to create different versions will require this parameter to be updates with the version you want to deploy