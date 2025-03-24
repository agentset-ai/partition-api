# modal serve .\main.py
from typing import Annotated
from fastapi import FastAPI, Header, status
from starlette.responses import JSONResponse
from pydantic import BaseModel
import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "llama-index>=0.12.23",
        "llama-index-readers-file>=0.4.6",
        "unstructured[all-docs]>=0.16.25",
        "modal>=0.73.91",
        "fastapi[standard]==0.115.11",
        "pydantic==2.9.2",
        "python-multipart>=0.0.20",
        "starlette==0.41.2",
        "requests>=2.31.0",
        "redis>=5.2.1",
    )
    .apt_install(
        "libmagic1",
        "poppler-utils",
        "tesseract-ocr",
        "tesseract-ocr-eng",
        "libreoffice",
        "pandoc",
        "libmagic-dev",
    )
)

app = modal.App(
    name="agentset-ingest",
    image=image,
    secrets=[modal.Secret.from_name("partitioner-secrets")],
)

web_app = FastAPI()


class IngestRequest(BaseModel):
    url: str | None = None
    text: str | None = None
    filename: str
    extra_metadata: dict | None = None
    unstructured_args: dict | None = None
    batch_size: int | None = None
    # tokenizer_model: str | None = None


@web_app.post("/ingest")
async def ingest(
    request: IngestRequest,
    api_key: Annotated[str | None, Header(alias="api-key")] = None,
):
    import os

    metadata = request.extra_metadata

    unstructured_args = {
        "strategy": "auto",  # fast, hi_res, ocr_only, auto
        "chunking_strategy": "basic",  # by_title, basic
    }
    if request.unstructured_args:
        unstructured_args.update(request.unstructured_args)

    if api_key != os.getenv("AGENTSET_API_KEY"):
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={
                "status": status.HTTP_401_UNAUTHORIZED,
                "message": "api-key is not valid!",
            },
        )

    # Count how many input sources are provided
    input_sources = sum(1 for x in [request.url, request.text] if x is not None)

    if input_sources == 0:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "status": status.HTTP_400_BAD_REQUEST,
                "message": "Either url or text must be provided",
            },
        )

    if input_sources > 1:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "status": status.HTTP_400_BAD_REQUEST,
                "message": "Only one of url or text can be provided",
            },
        )

    from io import BytesIO
    from llama_index.readers.file import UnstructuredReader

    # import tiktoken
    from unstructured.file_utils.filetype import detect_filetype
    import requests
    from redis import Redis
    import uuid
    import json

    file_stream = None
    filename_to_use = request.filename
    size_in_bytes = 0

    try:
        if request.url:
            try:
                response = requests.get(request.url)
                response.raise_for_status()
                size_in_bytes = len(response.content)
                file_stream = BytesIO(response.content)
            except Exception as e:
                return JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content={
                        "status": status.HTTP_400_BAD_REQUEST,
                        "message": f"Failed to download file from URL: {str(e)}",
                    },
                )
        else:  # text input
            text_bytes = request.text.encode("utf-8")
            size_in_bytes = len(text_bytes)
            file_stream = BytesIO(text_bytes)

        content_type = detect_filetype(
            file=file_stream,
            metadata_file_path=filename_to_use,
        ).mime_type
        documents = UnstructuredReader(
            allowed_metadata_types=(str, int, float, list, dict, type(None)),
        ).load_data(
            unstructured_kwargs={
                "file": file_stream,
                "metadata_filename": filename_to_use,
                "content_type": content_type,
                **unstructured_args,
            },
            split_documents=True,
            extra_info=metadata,
        )

        if len(documents) <= 0:
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "status": status.HTTP_500_INTERNAL_SERVER_ERROR,
                    "message": "couldn't parse document",
                },
            )

        result_documents = []
        total_characters = 0
        total_chunks = 0
        total_pages = None

        final_batch_size = request.batch_size or 5
        total_batches = 0
        # Initialize tokenizer from LlamaIndex
        # total_tokens = 0
        # tokenizer = tiktoken.encoding_for_model(
        #     request.tokenizer_model
        #     if request.tokenizer_model
        #     else "text-embedding-3-large"
        # )

        for document in documents:
            total_chunks += 1
            if document.text:
                text = document.text
                total_characters += len(text)
                # total_tokens += len(tokenizer.encode(text))
            if document.metadata.get("page_number"):
                total_pages = max(
                    total_pages if total_pages is not None else 0,
                    document.metadata.get("page_number"),
                )

            if (
                len(result_documents) == 0
                or len(result_documents[-1]) % final_batch_size == 0
            ):
                # Start new batch
                result_documents.append([document.to_dict()])
                total_batches += 1
            else:
                # Add to current batch
                result_documents[-1].append(document.to_dict())

        results_id = uuid.uuid4()
        batch_template = f"results_{results_id}_[BATCH_INDEX]"
        result = {
            "status": status.HTTP_200_OK,
            "metadata": {
                "filename": filename_to_use,
                "filetype": content_type,
                "sizeInBytes": size_in_bytes,
            },
            "total_characters": total_characters,
            "total_chunks": total_chunks,
            "total_batches": total_batches,
            "results_id": results_id,
            "batch_template": batch_template,
            # "total_tokens": total_tokens,
        }

        if total_pages is not None:
            result["total_pages"] = total_pages

        redis_client = Redis(
            host=os.getenv("REDIS_HOST"),
            port=os.getenv("REDIS_PORT"),
            password=os.getenv("REDIS_PASSWORD"),
            ssl=True,
        )

        # Store each batch in Redis with the specified key format
        for batch_idx, batch in enumerate(result_documents):
            redis_key = batch_template.replace("[BATCH_INDEX]", str(batch_idx))
            redis_client.set(redis_key, json.dumps(batch))

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=result,
        )
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "status": status.HTTP_500_INTERNAL_SERVER_ERROR,
                "message": str(e),
            },
        )


@app.function(timeout=7200)  # 2 hours
@modal.asgi_app()
def partition_api():
    return web_app
