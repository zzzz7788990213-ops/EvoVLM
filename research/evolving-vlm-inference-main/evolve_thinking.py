"""Run or resume the fixed Thinking local-Qwen evolution campaign."""

from evolution.dsl import THINKING_MODEL_ID
from evolution.local_campaign import cli_main


if __name__ == "__main__":
    raise SystemExit(cli_main(THINKING_MODEL_ID))
