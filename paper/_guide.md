---
created: 2024-09-24T14:04
updated: 2024-09-24T14:05
---
# How to use Cite in LaTeX code block

Since normal pandoc used `--citeproc` to process citation, we can NOT use \cite{ref} inside latex block code (started with \`\`\`{=latex}), instead we can use `\citeproc{ref-id}{ref-txt}`).

    ```{=latex}
        % example of using cite in latex code block
        As described in \citeproc{ref-cudatookit}{{[}32{]}}, the CUDA Toolkit provides a comprehensive development environment for creating high-performance GPU-accelerated applications.
    ```
**Note**: We <span style="color:red;"> must cite </span> the `id` in \citeproc{ref-id}{ref-txt} first (so it will be included in the reference list), then we can use \citeproc{ref-id}{ref-txt} later in the document.

# How to cite multiple references

You can use `[@ref1; @ref2; @ref3]` to cite multiple references in one citation.
