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

<!-- ! Requirements -->

# Software to build the paper

+ [Pandoc 3.3.1](https://github.com/jgm/pandoc/releases/tag/3.3)
+ [pandoc-crossref v0.3.17.1](https://github.com/lierdakil/pandoc-crossref/releases/tag/v0.3.17.1c) git commit 56c14dcf687efcdaed37a9ceff3abd39ee0067a8 (HEAD) built with Pandoc v3.3, pandoc-types v1.23.1 and GHC 9.6.5
+ [Textlive 2022- for windows](https://www.tug.org/texlive/) - but for WSL, just install newest textlive 2026

## How to install TeX Live 2026 in WSL
### 1. Mount the ISO

```bash
sudo mkdir -p /mnt/texlive
sudo mount -o loop texlive.iso /mnt/texlive
```

### 2. Run the Installer from Local Mount

```bash
sudo /mnt/texlive/install-tl
```

This reads entirely from your local disk — **no network required**, so it will be very fast. The same installer menu appears as before. Press `I` to start.

### 3. Update `.bashrc` with 2026 Paths

After installation finishes, update the year in your PATH:

```bash
nano ~/.bashrc
```

Add at the bottom:
```bash
# TeX Live 2026 PATH settings
export PATH=/usr/local/texlive/2026/bin/x86_64-linux:$PATH
export MANPATH=/usr/local/texlive/2026/texmf-dist/doc/man:$MANPATH
export INFOPATH=/usr/local/texlive/2026/texmf-dist/doc/info:$INFOPATH
```

Then reload:
```bash
source ~/.bashrc
```

### 4. Unmount ISO After Installation

```bash
sudo umount /mnt/texlive
```

### 5. Verify

```bash
tex --version
# Expected: TeX 3.141592653 (TeX Live 2026)
```

Then after this, you only need to install the matching **Pandoc + pandoc-crossref** versions for TL2026 and your setup will be complete.

## Update PATH for TeX Live 2025 in WSL
```bash
# TeX Live 2025 PATH settings
export PATH=/usr/local/texlive/2025/bin/x86_64-linux:$PATH
export MANPATH=/usr/local/texlive/2025/texmf-dist/doc/man:$MANPATH
export INFOPATH=/usr/local/texlive/2025/texmf-dist/doc/info:$INFOPATH
```
