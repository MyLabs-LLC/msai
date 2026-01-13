In this project, you are going to make a chatbot to scrape LLM Inference Serving websites to research costs of serving various LLMs. You will do this by writing an MCP Server that hooks up to Firecrawl's API and saving the data in a SQLite Database. You should use the following websites to scrape:

- "cloudrift": "https://www.cloudrift.ai/inference"
- "deepinfra": "https://deepinfra.com/pricing"
- "fireworks": "https://fireworks.ai/pricing#serverless-pricing"
- "groq": "https://groq.com/pricing"

### Setup (conda base + pip)

1. Ensure you have **Python 3.10+** available (conda base is fine).
2. Install dependencies from the repo-wide requirements file:

```bash
pip install -r /home/lence/msai/requirements.txt
```

3. Set your API keys.

**Create a `.env` file** in the `PriceScout/` directory with the following content:

```bash
# Create .env file
cd /home/lence/msai/PriceScout
cat > .env << 'EOF'
# Anthropic Claude API Key
# For Vocareum: Use your voc-* key (base URL will be auto-detected)
ANTHROPIC_API_KEY=voc-75828985118751220613916942103b0d3fc0.08126595

# Optional: Override Anthropic API base URL
# If using Vocareum, this will be auto-set to https://claude.vocareum.com
# Only set this if you need to override the default behavior
# ANTHROPIC_BASE_URL=https://claude.vocareum.com

# Firecrawl API Key for web scraping
# Get your key from: https://firecrawl.dev
FIRECRAWL_API_KEY=your-firecrawl-api-key-here
EOF
```

**Or manually create `.env` file** with:
```
ANTHROPIC_API_KEY=voc-75828985118751220613916942103b0d3fc0.08126595
FIRECRAWL_API_KEY=your-firecrawl-api-key-here
```

**Note:** If you're using a **Vocareum** Claude key (it starts with `voc-`), the client will automatically route requests to `https://claude.vocareum.com` even if you don't set `ANTHROPIC_BASE_URL`, but setting it explicitly is fine.

**Alternative:** If `.env` files are blocked in your workspace, you can export variables in your shell:
```bash
export ANTHROPIC_API_KEY="voc-75828985118751220613916942103b0d3fc0.08126595"
export FIRECRAWL_API_KEY="your-firecrawl-api-key-here"
```

4. Complete the TODO sections in:
   - `PriceScout/starter_server.py` (the MCP scraper tools)
   - `PriceScout/starter_client.py` (the MCP client/orchestrator)

5. Confirm `PriceScout/server_config.json` is pointing at `starter_server.py` and your SQLite DB path (this repo’s version is already configured for the workspace path).
6. Run the client:

```bash
python /home/lence/msai/PriceScout/starter_client.py
```

7. Note: the filesystem MCP server uses `npx`, so you’ll need **Node.js** installed for that tool to work.

### Example prompts

Use the following prompts in your chatbot, but play around with all the LLM providers in the list above:
    - "How much does cloudrift ai (https://www.cloudrift.ai/inference) charge for deepseek v3?"
    - "How much does deepinfra (https://deepinfra.com/pricing) charge for deepseek v3"
    - "Compare cloudrift ai and deepinfra's costs for deepseek v3"
