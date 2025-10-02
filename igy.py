!pip install sagemaker --upgrade
!pip install requests
#Set up the session and deploy the model
import sagemaker 
import boto3
import json
import base64
from sagemaker.jumpstart.model import JumpStartModel
from PIL import Image
import requests
from io import BytesIO
import time
# 1. Set up the SageMaker session and role
print("Step 1: Setting up SageMaker session and IAM role...")
sagemaker_session = sagemaker.Session()
aws_region = sagemaker_session.boto_region_name
role = sagemaker.get_execution_role()
print(f"SageMaker session established in region: {aws_region}")
print(f"Using IAM role: {role}")
# 2. Deploy a pre-trained model from SageMaker
JumpStart
# This example uses a TensorFlow MobileNetV2 model for image classification.
# For other models, you can find the model_id on the SageMaker JumpStart documentation.
model_id = "tensorflow-ic-mobilenet-v2-100-224-classification"
endpoint_name = f'jumpstart-{model_id}-endpoint'
# Create a JumpStartModel object
model = JumpStartModel(model_id=model_id)
print(f"\nStep 2: Deploying model '{model_id}' to a SageMaker endpoint...")
predictor = model.deploy(
 initial_instance_count=1,
 instance_type='ml.m5.xlarge', # Adjust instance type if needed
 endpoint_name=endpoint_name,
 wait=True # Wait for the deployment to complete
)
print(f"Endpoint '{predictor.endpoint_name}' deployed successfully.")
# 3. Get a test image and preprocess it
print("\nStep 3: Preparing a test image for inference...")
image_url = "https://hips.hearstapps.com/hmg-prod/images/dog-puppy-on-garden-royalty-free-image1586968018.jpg"
try:
 response = requests.get(image_url)
 image = Image.open(BytesIO(response.content))
 print("Test image downloaded successfully.")
except Exception as e:
 print(f"Error downloading image: {e}")
 raise
# The MobileNetV2 model expects input as a JSON object containing the base64-encoded image.
# We resize the image to the model's expected dimensions (224x224).
resized_image = image.resize((224, 224))
image_bytes = BytesIO()
resized_image.save(image_bytes, format="JPEG")
image_bytes = image_bytes.getvalue()
image_b64 = base64.b64encode(image_bytes).decode('utf-8')
payload = {"instances": [{"b64": image_b64}]}
# 4. Invoke the endpoint and get the prediction
print("\nStep 4: Invoking the endpoint with the test image...")
try:
 prediction = predictor.predict(payload)

 # The output format depends on the model. This model returns a dictionary.
 if prediction and 'predictions' in prediction:
 top_prediction = prediction['predictions']

 print("\nPrediction received:")
 # Pretty-print the JSON output
 print(json.dumps(top_prediction, indent=2))

 # Extract and print the top predicted class index and confidence.
 predicted_index = max(top_prediction, key=top_prediction.get)
 confidence = top_prediction[predicted_index]
 print(f"\nTop Predicted Class Index: {predicted_index}")
 print(f"Confidence: {confidence:.4f}")
 else:
 print("Error: Invalid prediction response format.")
 print(f"Full response: {prediction}")
except Exception as e:
 print(f"An error occurred while invoking the endpoint: {e}")
Clean up resources
# 5. Clean up resources
print("\nStep 5: Deleting the SageMaker endpoint to clean up resources...")
try:
 predictor.delete_endpoint()
 print(f"Endpoint '{predictor.endpoint_name}' deleted successfully.")
except Exception as e:
 print(f"An error occurred while deleting the endpoint: {e}")
