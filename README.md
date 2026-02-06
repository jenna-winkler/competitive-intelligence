# Competitive Intelligence Agent

A competitive intelligence research agent built with BeeAI Framework and Agent Stack.

## Requirements

- Agent Stack installed and running ([quickstart](https://agentstack.beeai.dev/stable/introduction/quickstart))
- OpenAI API key

## Setup

Clone the repo:

```bash
git clone https://github.com/jenna-winkler/competitive-intelligence.git
cd competitive-intelligence
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
uv sync
```

Create a .env file from the template and set your API key:

```bash
cp .env.template .env

OPENAI_API_KEY=your_api_key_here
```

## Run

Start the agent server:

```bash
uv run server
```

## Example prompts

* What emerging trends in agentic AI platforms should enterprises prepare for?
* Compare OpenAI, Anthropic, and Google’s agent strategies
* How does Microsoft’s AI strategy compare to AWS?
