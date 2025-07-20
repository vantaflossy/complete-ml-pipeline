# End to End Complete ML Pipeline
This project covers the end to end understanding for creating an ML pipeline and working around it using DVC for experiment, tracking, and data versioning using AWS S3

## 🚀 Overview
This project demonstrates a production-ready ML pipeline that includes:

Data versioning with DVC and AWS S3
Experiment tracking and reproducibility
Pipeline orchestration with automated stages
Model versioning and deployment readiness
Collaborative workflow for ML teams

## 🛠️ Prerequisites

- Python 3.8+ 
- AWS CLI configured with appropriate permissions
- Git
- DVC
  
## Getting Started
1. Clone this repository:
   
   ```
   git clone https://github.com/vantaflossy/complete-ml-pipeline
   cd complete-ml-pipeline
   ```
2. Install necessary dependencies:

   ```
   pip install -r requirements.txt
   ```
3. Run the pipeline using `dvc repro`. Create and track experiments using `dvc exp run`. Keep changing the values in `params.yaml` to find the best result. Check the result in `/metrics/evaluation_metrics.json`

4. Configure AWS using `aws configure` and entering your IAM user details. Then add your AWS user to dvc using

   ```
   dvc remote add -d <name> s3://<your-bucket-name>
   ```

## Contributions:
You are welcome to contribute to this project. Send a pull request and I will review it accordingly!


