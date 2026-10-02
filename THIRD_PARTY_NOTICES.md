# Third-party notices

Public availability of this repository does not replace the licenses or rights
associated with individual upstream materials. No new blanket license is
assigned to the whole repository in this release.

## CharXiv

The `research/evolving-vlm-inference-main/charxiv/` directory contains material
from [CharXiv](https://github.com/princeton-nlp/CharXiv). Its original README,
attributions, and paper citation are retained.

- Upstream code: Apache License 2.0. The upstream license text is included at
  [`charxiv/LICENSE`](research/evolving-vlm-inference-main/charxiv/LICENSE).
- Original dataset contributions other than chart images: Creative Commons
  Attribution-ShareAlike 4.0 International. The upstream license text is included
  at [`charxiv/data/LICENSE`](research/evolving-vlm-inference-main/charxiv/data/LICENSE).
  Derived query subsets and annotation records retain this attribution and
  applicable license. Their selection and arrangement are documented by the
  accompanying data manifests.
- Chart images: rights remain with the original chart authors. Original sources
  are listed in the upstream image metadata files under `charxiv/data/`. Neither
  of the licenses above is asserted as a blanket license for those images.

The two license texts were restored from the upstream repository for this
publication; source URLs and hashes are recorded in
[`PUBLICATION_PROVENANCE.json`](PUBLICATION_PROVENANCE.json).

## Qwen

Inference depends on the external
[Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) and
[Qwen3-VL-2B-Thinking](https://huggingface.co/Qwen/Qwen3-VL-2B-Thinking)
snapshots. Model weights are not redistributed here. Obtain and use each
snapshot under its own model-card and license terms.

## Report template and software dependencies

The archived report includes the upstream NeurIPS 2025 style file with its
existing notices and provenance. This is a formatting dependency, not a claim
of venue acceptance. Python dependencies retain their respective licenses.

Upstream citations identify the sources used by the project. They are not
claims of authorship, institutional affiliation, collaboration, or endorsement
of EvoVLM by upstream contributors.
