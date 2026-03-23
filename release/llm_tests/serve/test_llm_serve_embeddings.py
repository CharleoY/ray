"""Release test for Ray Serve LLM embeddings endpoint.

This test validates that the embeddings endpoint works correctly with
pooling models (e.g., Qwen3-8B-emb) using runner="pooling" and convert="embed".
"""

import pytest
from openai import OpenAI
from ray import serve
from ray.serve.llm import LLMConfig, ModelLoadingConfig, build_openai_app

from test_utils import create_openai_client, wait_for_server_ready

# Using a small embedding model for testing
MODEL_ID = "BAAI/bge-small-en-v1.5"
RAY_MODEL_ID = "bge-small-en"


def get_llm_config() -> LLMConfig:
    """Create LLMConfig for embedding model."""
    return LLMConfig(
        model_loading_config=ModelLoadingConfig(
            model_id=RAY_MODEL_ID,
            model_source=MODEL_ID,
        ),
        deployment_config=dict(
            autoscaling_config=dict(
                min_replicas=1,
                max_replicas=1,
            ),
        ),
        engine_kwargs=dict(
            task="embed",
        ),
        runtime_env=None,
    )


def start_ray_serve() -> str:
    """Start Ray Serve with embedding model."""
    ray_url = "http://localhost:8000"
    llm_config: LLMConfig = get_llm_config()
    app = build_openai_app({"llm_configs": [llm_config]})
    serve.run(app, blocking=False)
    return ray_url


def generate_embedding(client: OpenAI, model_id: str, text: str) -> list:
    """Generate embedding using the provided OpenAI client."""
    response = client.embeddings.create(
        model=model_id,
        input=text,
    )
    return response.data[0].embedding


def test_llm_serve_embeddings():
    """Test that Ray Serve LLM embeddings endpoint works correctly."""
    test_text = "This is a test sentence for embedding."

    print("Starting Ray Serve LLM with embedding model")
    ray_url = start_ray_serve()
    ray_client = create_openai_client(ray_url)

    wait_for_server_ready(ray_url, model_id=RAY_MODEL_ID, timeout=240)

    print("Generating embedding...")
    embedding = generate_embedding(ray_client, RAY_MODEL_ID, test_text)

    # Validate embedding response
    assert embedding is not None, "Embedding should not be None"
    assert isinstance(embedding, list), "Embedding should be a list"
    assert len(embedding) > 0, "Embedding should not be empty"
    assert all(
        isinstance(x, (int, float)) for x in embedding
    ), "All elements should be numeric"

    print(f"Successfully generated embedding with dimension: {len(embedding)}")

    serve.shutdown()


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
