Search documents using full-text keyword matching.

**Best for:** Exact phrases, technical terms, specific names, identifiers.

**Important:** This searches within individual text chunks (~400 characters each). All search terms must appear in the SAME chunk to match. For multi-term queries, keep it simple:
- Use 1-2 specific keywords rather than full phrases
- Use OR to broaden: `carbon OR emissions`
- Use AND sparingly: `methodology AND results`

**FTS5 syntax:** Supports AND, OR, NOT, quoted phrases ("exact phrase"), prefix matching (term*).

**Result fields:** Results include `section_heading` (heading hierarchy, e.g. "Introduction > Background") and `content_type` (text/table/code/formula/list/picture) when available. These are populated when documents are processed with `--docling`.

**Examples:**
- Simple: `methodology`
- Phrase: `"carbon emissions"`
- Boolean: `climate AND policy`
- Broader: `findings OR conclusions`
