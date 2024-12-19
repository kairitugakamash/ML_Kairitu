import sagemaker
from sagemaker import get_execution_role

# Step 2: Define S3 Bucket and Roles

sagemaker_session = sagemaker.Session()

# Get a SageMaker-compatible role used by this Notebook Instance.
role = get_execution_role()

bucket = sagemaker_session.default_bucket()
prefix = 'clip-model'

model_location = sagemaker_session.upload_data(
    'model_path', 
    bucket=bucket, 
    key_prefix=prefix
)

# Step 3.5: Verify the Model in S3 with boto3
import boto3

s3 =
# Next, let's list all objects in our S3 bucket:

boto3.client('s3')

response = s3.list_objects(Bucket=bucket)

for content in response['Contents']:
    print(content['Key'])
    
# Step 4: Create a Model
from sagemaker.model import Model

clip_model = Model(
    model_data=model_location,
    imagker-image',
    role=role
)

# Step 5: Deploy the Model for Real-Time Inference

clip_predictor = clip_model.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.large',
    endpoint_name='clip-endpoint'
)

# Step 6: Create a Predictor from an Existing Deployment
from sagemaker.predictor import Predictor

endpoint_name = "huggingface-pytorch-inference-2023-03-18-13-33-18-657"  
# Existing endpoint
clip_predictor = Predictor(endpoint_name)

#Making Inferences
import requests
from PIL import Image
import numpy as np
import json

url = "http://images.cocodataset.org/val2017/000000039769.jpg"

image = Image.open(requests.get(url, stream=True).raw)
image_array = np.array(image)

data = {
  "inputs": "the mesmerizing performances of the leads keep the film grounded and keep the audience riveted.",
  "pixel_values": image_array.tolist()
}

response = clip_predictor.predict(json.dumps(data))
print(response)

# Step 7: Offline Inference with Batch Transform
clip_transformer = clip_model.transformer(
    instance_count=1,
    instance_type='ml.m5.large',
    strategy='SingleRecord',
    assemble_with='Line',
    output_path='s3://{}/{}/output'.format(bucket, prefix)
)

# Then, start a transform job:
clip_transformer.transform(
    data='s3://{}/{}/input'.format(bucket, prefix),
    content_type='application/x-image',
    split_type='None'
)
clip_transformer.wait()
