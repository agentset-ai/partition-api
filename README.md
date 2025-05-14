<h3 align="center">Agentset - Partition API</h3>

<br/>

This is the partition API used by the [Agentset Platform](https://github.com/agentset-ai/agentset).

## Tech Stack

- [Modal](https://modal.com/) – deployments
- [Unstructured](https://unstructured.io/) – document parsing
- [LlamaIndex](https://www.llamaindex.ai/) chunking
- [FastAPI](https://fastapi.tiangolo.com/) – API

## Deployment

1. Install dependencies:

```bash
uv sync
```

2. Create your modal API token (Manage Workspaces -> API Tokens)

3. Link modal cli to your account (copy the commands from the modal dashboard and run them with `uv run`):

```bash
uv run modal token set --token-id ak-xxx --token-secret as-xxx --profile=example
```

```bash
uv run modal profile activate example
```

4. Create secrets (read `.env.example` for more info about the variables):

```bash
uv run modal secret create --force partitioner-secrets QSTASH_URL=https://qstash.io QSTASH_TOKEN=xxx REDIS_HOST=xxx REDIS_PORT=xxx REDIS_PASSWORD=xxx AGENTSET_API_KEY=xxx
```

5. Deploy the app:

```bash
uv run modal deploy main.py
```

6. Done 🎉
   <br />

Now take the URL from the modal dashboard `PARTITION_API_URL` and the `AGENTSET_API_KEY` you specified above, and add them to the [Agentset Platform](https://github.com/agentset-ai/agentset) as `PARTITION_API_URL` and `PARTITION_API_KEY`, respectively.

They should look like this:

```bash
PARTITION_API_URL=https://example.modal.run/ingest
PARTITION_API_KEY=xxx
```

## License

This project is open-source under the MIT License. You can [find it here](https://github.com/agentset-ai/partition-api/blob/main/LICENSE.md).
