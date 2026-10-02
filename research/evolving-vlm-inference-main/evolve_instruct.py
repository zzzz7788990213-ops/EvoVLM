"""Run or resume the fixed Instruct local-Qwen evolution campaign."""

from evolution.dsl import INSTRUCT_MODEL_ID
from evolution.local_campaign import cli_main


if __name__ == "__main__":
    raise SystemExit(cli_main(INSTRUCT_MODEL_ID))
