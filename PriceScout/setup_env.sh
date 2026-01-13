#!/bin/bash
# Helper script to create .env file for PriceScout project

ENV_FILE=".env"

if [ -f "$ENV_FILE" ]; then
    echo "⚠️  .env file already exists. Backing up to .env.backup"
    cp "$ENV_FILE" "${ENV_FILE}.backup"
fi

cat > "$ENV_FILE" << 'EOF'
# PriceScout MCP Project - Environment Variables
# This file is loaded automatically by python-dotenv

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

echo "✅ Created $ENV_FILE file"
echo "📝 Please edit $ENV_FILE and add your FIRECRAWL_API_KEY"
