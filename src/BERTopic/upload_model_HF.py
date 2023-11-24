import argparse
from huggingface_hub import login
from bertopic import BERTopic
import os

def main(model_path, repo_id):
    # Retrieve the access token for Hugging Face from environment variables
    access_token_write = os.environ.get("ACCESS_TOKEN_HF")

    # Login to Hugging Face using the access token
    login(token=access_token_write)

    # Load the BERTopic model from the specified path
    topic_model = BERTopic.load(model_path)

    # Push the model to Hugging Face Hub
    topic_model.push_to_hf_hub(
        repo_id=repo_id,
        save_ctfidf=True,
        serialization='safetensors',
        save_embedding_model='paraphrase-multilingual-MiniLM-L12-v2'
    )

if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description="Upload BERTopic model to Hugging Face Hub")

    # Add arguments
    parser.add_argument("--path", help="Path to your BERTopic model folder", required=True)
    parser.add_argument("--HF_repo_ID", help="Hugging Face repository ID", required=True)

    # Parse the arguments
    args = parser.parse_args()

    # Call the main function with parsed arguments
    main(args.path, args.HF_repo_ID)


