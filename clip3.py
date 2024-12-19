!pip install transformers
!pip install accelerate
!pip install pillow

import sagemaker
from sagemaker.huggingface import HuggingFaceModel
from sagemaker import get_execution_role

# Step 2: Define S3 Bucket and Roles
role = get_execution_role()

model_data = "s3://bucket/model.tar.gz"

huggingface_model = HuggingFaceModel(
    model_data = model_location,
    role = role,
    transformers_verion = "4.26",
    pytorch = "1.13",
    py_version = "py39"
)


predictor = huggingface_model.deploy(
    initial_instance_count=1,
    instance_type="ml.g5.2xlarge"

)

from PIL import Image
from io import BytesIO
import base64

image = Image.open("image.tif")
image = image.resize((1600,1600))
buffered = BytesIO()

image.save(buffered, format='JPEG')
base64_bytes = base64.b64encode(buffered.getvalue())

import boto3
import json
import base64

sagemaker_client = boto3.client("runtime.sagemaker")

base64_encoded = base64_bytes.decode("utf-8")

# if you intend to use the sagemaker endpoint

request_body = {
    "inputs": base64_encoded,
    "parameters": {
        "candidate_labels": ['cloud', 'terrain', 'noise']
    }
}

request_body_json = json.dumps(request_body)

response = sagemaker_client.invoke_endpoint(
    EndpointName = "endpointName",
    contentType = "application/json",
    Body=request_body_json
)

    
response_body = json.loads(response['Body'].read().decode())

response_body
