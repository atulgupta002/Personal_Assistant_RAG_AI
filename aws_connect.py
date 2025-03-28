import boto3
import json
from semantic_search import build_llm_prompt
import re
import logging
from datetime import datetime

logger = logging.getLogger()
logger.setLevel(logging.INFO)
BUCKET_NAME = 'personal-assistant-app-bucket'
s3 = boto3.client('s3',region_name='us-east-1')

query_prompt = "static/query_prompt.txt"
greeting_prompt = "static/greeting_prompt.txt"

# Function to log queries and results in S3.
def log_query_response(query, response):
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    filename = f"logs/query_response_{timestamp}.txt"
    
    log_content = f"Query: {query}\nResponse: {response}\n\n"
    
    try:
        s3.put_object(
            Bucket=BUCKET_NAME,
            Key=filename,
            Body=log_content
        )
        logger.info(f"Logged query and response to S3: {filename}")
    except Exception as e:
        logger.error(f"Error logging query/response to S3: {str(e)}")

# Function to determine if a query is a greeting or otherwise.
def detect_intent(query):
    query = query.strip().lower()
    
    # Check for standalone greetings
    if query in {'hi', 'hello', 'hey', 'hi!', 'hello!'}:
        return "greeting"
        
    # Check for greeting patterns
    greeting_phrases = {
        'good morning', 'good afternoon', 'good evening',
        'how are you', "what's up", 'howdy', 'yo'
    }
    
    if any(phrase in query for phrase in greeting_phrases):
        return "greeting"
    else:
        return "context_query"

# Function to generate and get ai response using AWS Bedrock. It is then return to Flask application.
def get_ai_response(user_input, conversation_history):
    bedrock_client = boto3.client('bedrock-runtime', region_name='us-east-1')

    intent = detect_intent(user_input)

    if intent == "greeting":
        prompt = build_llm_prompt(user_input,greeting_prompt,intent)
    else:
        prompt = build_llm_prompt(user_input,query_prompt,intent)
    
    response = bedrock_client.invoke_model(
    modelId="anthropic.claude-v2",
    body=json.dumps({"prompt": prompt, "max_tokens_to_sample": 200})
    )

    response_body = json.loads(response["body"].read())
    log_query_response(user_input, response_body["completion"])
    return response_body["completion"]
