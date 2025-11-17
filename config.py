import os
import boto3
from botocore.config import Config
from dotenv import load_dotenv
from langchain_aws import BedrockEmbeddings, ChatBedrock

load_dotenv()

def initialize_bedrock_client():
    """Initialize AWS Bedrock client."""
    config = Config(retries={'max_attempts': 3, 'mode': 'adaptive'})
    return boto3.client(
        service_name="bedrock-runtime",
        region_name=os.getenv("AWS_REGION", ""),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", ""),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", ""),
        config=config
    )

def initialize_embeddings(client):
    """Initialize Bedrock embeddings model."""
    return BedrockEmbeddings(client=client, model_id="amazon.titan-embed-text-v1")

def initialize_llm(client):
    """Initialize Bedrock LLM."""
    return ChatBedrock(
        client=client,
        model_id="anthropic.claude-3-haiku-20240307-v1:0",
        model_kwargs={
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 2000,
            "temperature": 0.7
        }
    )
