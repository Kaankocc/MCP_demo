# main.py

import asyncio
from rag_agent import run_parallel_agent

async def main():
    await run_parallel_agent()

if __name__ == "__main__":
    asyncio.run(main())
