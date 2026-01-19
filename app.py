import re
from typing import List, Tuple, Dict

import streamlit as st
from anthropic import Anthropic

# Try to import typed errors (available in newer anthropic SDKs)
try:
    from anthropic import NotFoundError, AuthenticationError, RateLimitError, APIError  # type: ignore
except Exception:  # pragma: no cover
    NotFoundError = AuthenticationError = RateLimitError = APIError = Exception  # fallback


# ---------------------------
# Helpers
# ---------------------------
def parse_excel_paste(pasted: str) -> List[str]:
    """
    Accepts text copied from Excel (often tab-separated columns per row) or plain lines.
    Returns a clean list of answer strings.
    Heuristic: for tab-separated rows, pick the *longest* cell as the likely answer.
    """
    answers: List[str] = []
    for raw_line in (pasted or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if "\t" in line:
            cells = [c.strip() for c in line.split("\t") if c.strip()]
            if not cells:
                continue
            candidate = max(cells, key=len)
            if len(candidate) < 5 and len(cells) >= 2:
                candidate = cells[-1]
            line = candidate.strip()

        lower = line.lower()
        if lower in {"answer", "answers", "response", "responses", "svar", "kommentar", "comments"}:
            continue

        answers.append(line)

    return answers


def downsample_evenly(items: List[str], max_n: int) -> Tuple[List[str], bool]:
    if max_n <= 0 or len(items) <= max_n:
        return items, False

    n = len(items)
    if max_n == 1:
        return [items[0]], True

    indices = [round(i * (n - 1) / (max_n - 1)) for i in range(max_n)]
    seen = set()
    sampled = []
    for idx in indices:
        if idx not in seen:
            sampled.append(items[idx])
            seen.add(idx)

    return sampled, True


def extract_text_from_anthropic_message(message) -> str:
    parts = []
    for block in getattr(message, "content", []) or []:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    return "\n".join(parts).strip()


def strip_code_fences(md: str) -> str:
    fenced = re.match(r"^\s*```(?:\w+)?\s*(.*?)\s*```\s*$", md, flags=re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    md = re.sub(r"(?m)^\s*```.*?$", "", md)
    return md.strip()


def looks_like_valid_output(md: str, n_articles: int) -> bool:
    if not md or not md.strip():
        return False
    if "```" in md:
        return False

    for i in range(1, n_articles + 1):
        if not re.search(rf"(?m)^##\s*Article\s*{i}\b", md, flags=re.IGNORECASE):
            return False
    return True


def split_into_n_articles(md: str, n_articles: int) -> Tuple[List[str], bool]:
    """
    Split output into N Markdown strings based on headings:
    ## Article 1 ...
    ## Article 2 ...
    ...
    Returns ([a1..aN], success_flag)
    """
    pattern = r"(?m)^##\s*Article\s*(\d+)\b.*$"
    matches = list(re.finditer(pattern, md, flags=re.IGNORECASE | re.MULTILINE))
    if not matches:
        return [md.strip()], False

    # Find first occurrence start for each article number
    starts: Dict[int, int] = {}
    for m in matches:
        try:
            num = int(m.group(1))
        except Exception:
            continue
        if 1 <= num <= n_articles and num not in starts:
            starts[num] = m.start()

    if not all(i in starts for i in range(1, n_articles + 1)):
        return [md.strip()], False

    ordered = sorted(starts.items(), key=lambda x: x[1])  # [(num, start), ...]
    slices: Dict[int, str] = {}

    for idx, (num, start) in enumerate(ordered):
        end = ordered[idx + 1][1] if idx + 1 < len(ordered) else len(md)
        slices[num] = md[start:end].strip()

    return [slices[i] for i in range(1, n_articles + 1)], True


def build_prompt(question: str, answers: List[str], optional_context: str, n_articles: int) -> str:
    answers_block = "\n".join([f"{i+1}. {a}" for i, a in enumerate(answers)])
    ctx = (optional_context or "").strip()
    ctx_block = ctx if ctx else "No extra context provided."

    # Build exact required structure for N articles
    structure_lines = []
    for i in range(1, n_articles + 1):
        structure_lines.append(f"## Article {i} — <Angle title>\n<200–400 words>\n")
    structure_block = "\n".join(structure_lines).strip()

    return f"""
You are writing as Opticom: a leading consultancy in the forest industry known for high-quality stakeholder dialogues and close contact with decision-makers across the value chain.

TASK
Based on patterns across many anonymised interview answers, write {n_articles} distinct LinkedIn articles.

CONFIDENTIALITY (CRITICAL)
- Do NOT mention any client, brand, mill, retailer, or company names.
- Do NOT reference specific projects, contract details, or anything that could identify a single account.
- Do NOT write “a client told us X” or “our customer said…”.
- Use aggregated language such as “Across many conversations…” / “A recurring pattern is…”.
- Keep it concrete enough to feel real, but abstract enough to protect confidentiality.
- Avoid verbatim quotes from the input.

VOICE & POSITIONING (IMPORTANT)
- Write from Opticom’s perspective (“we”), but stay humble and down to earth.
- Subtly signal that we are close to customers’ customers, able to spot patterns others miss, and focused on forward-looking issues.
- No sales language. No CTA like “contact us” / “book a call”.
- Credibility should come from insight, not promotion.

INPUTS
Interview question:
{question.strip()}

Optional context (may guide framing; do not invent facts beyond the answers):
{ctx_block}

Interview answers (raw, anonymised):
{answers_block}

OUTPUT REQUIREMENTS (STRICT)
- Output MARKDOWN ONLY.
- Output EXACTLY in this structure (no extra sections before/after):

{structure_block}

STYLE (LinkedIn-ready)
- Strong opening hook (1–2 short paragraphs).
- Scannable formatting: short paragraphs; optional bullet list (max 4 bullets) if helpful.
- Thought-leader tone, but grounded and plain-spoken.
- Clear takeaway at the end (1–2 sentences).

QUALITY BAR
Each article must have a clearly different:
- angle (e.g., risk, strategy, operating model, market dynamics, innovation tension, credibility of claims, procurement logic),
- narrative arc (hook → insight → implications → takeaway),
- emphasis (choose different themes across the set).
""".strip()


def call_claude_markdown(prompt: str, model: str, max_tokens: int, temperature: float) -> str:
    client = Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system="You output Markdown only. Never wrap the output in code fences.",
        messages=[{"role": "user", "content": prompt}],
    )
    return extract_text_from_anthropic_message(message)


def friendly_model_error(model_id: str) -> str:
    return (
        f"Model not found: **{model_id}**\n\n"
        "Fix: pick a valid Anthropic model ID in **1. Context → Model**.\n"
        "Recommended defaults:\n"
        "- `claude-sonnet-4-5-20250929` (stable snapshot)\n"
        "- `claude-sonnet-4-5` (alias)\n"
        "- `claude-haiku-4-5` (faster/cheaper)\n"
        "- `claude-opus-4-5` (premium)\n"
    )


# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="NewsCoder", layout="wide")
st.title("NewsCoder")

st.write(
    "Paste **one interview question** and **many answers**, and generate **1–5 distinct LinkedIn articles**. "
    "The output is insight-driven, anonymised, and written from **Opticom’s perspective** — thought-leader, but down to earth."
)

MODEL_PRESETS = {
    "Claude Sonnet 4.5 — stable snapshot (recommended)": "claude-sonnet-4-5-20250929",
    "Claude Sonnet 4.5 — alias": "claude-sonnet-4-5",
    "Claude Haiku 4.5 — alias (fast/cheap)": "claude-haiku-4-5",
    "Claude Opus 4.5 — alias (premium)": "claude-opus-4-5",
    "Custom (type your own model ID)": "__custom__",
}

with st.expander("1. Context", expanded=True):
    st.caption("Optional framing. Helpful if you want a specific emphasis, but you can leave it empty.")

    optional_context = st.text_area(
        "Optional context (e.g., what this question was about, what to emphasise, audience nuance)",
        placeholder="Example: This question touches on supply reliability, credibility of sustainability claims, and how purchasing decisions are actually made across the value chain.",
        height=120,
    )

    col_a, col_b, col_c, col_d = st.columns([1.2, 0.8, 1, 1])
    with col_a:
        model_choice = st.selectbox("Model", list(MODEL_PRESETS.keys()), index=0)
        custom_model = ""
        if MODEL_PRESETS[model_choice] == "__custom__":
            custom_model = st.text_input("Custom model ID", placeholder="e.g. claude-sonnet-4-5-20250929")

        model_id = custom_model.strip() if custom_model.strip() else MODEL_PRESETS[model_choice]
        if model_id == "__custom__":
            model_id = ""  # force validation

    with col_b:
        n_articles = st.selectbox("Number of LinkedIn articles", options=[1, 2, 3, 4, 5], index=2)

    with col_c:
        max_answers_to_send = st.number_input(
            "Max answers to send to Claude",
            min_value=20,
            max_value=500,
            value=160,
            step=10,
        )

    with col_d:
        temperature = st.slider("Creativity (temperature)", min_value=0.0, max_value=1.0, value=0.6, step=0.05)
        st.caption("Lower = more conservative and consistent. Higher = more varied angles and phrasing.")

with st.expander("2. Paste Input", expanded=True):
    st.caption("Paste directly from Excel. If a row has multiple columns, the app picks the longest cell as the answer.")
    question = st.text_input("Interview question", placeholder="Paste the interview question here…")

    st.markdown("**Paste answers (copied from Excel)**")
    pasted_answers = st.text_area(
        "Answers",
        placeholder=(
            "Example (copied from Excel):\n"
            "RESP_001\tWe struggle with lead time volatility...\n"
            "RESP_002\tOur customers ask for proof behind claims...\n"
            "RESP_003\tPrice matters, but reliability matters more...\n"
        ),
        height=240,
        label_visibility="collapsed",
    )

generate = st.button("Generate", type="primary", use_container_width=True)

if "result_md" not in st.session_state:
    st.session_state.result_md = ""
    st.session_state.articles = []
    st.session_state.parse_info = ""
    st.session_state.n_articles_last = 3

if generate:
    # --- Validation
    if not model_id or not model_id.strip():
        st.error("Please select a model (or type a custom model ID).")
        st.stop()

    if not question or not question.strip():
        st.error("Please add the interview question.")
        st.stop()

    if not pasted_answers or not pasted_answers.strip():
        st.error("Please paste the interview answers (copied from Excel).")
        st.stop()

    # --- Parse answers
    answers = parse_excel_paste(pasted_answers)
    if len(answers) < 5:
        st.error("I found fewer than 5 answers after parsing. Please check your pasted input.")
        st.stop()

    sampled_answers, was_sampled = downsample_evenly(answers, int(max_answers_to_send))
    info = f"Parsed **{len(answers)}** answers."
    if was_sampled:
        info += f" Sent an evenly-sampled subset of **{len(sampled_answers)}** answers to Claude (to keep prompts manageable)."
    st.session_state.parse_info = info

    # --- Build prompt
    prompt = build_prompt(
        question=question,
        answers=sampled_answers,
        optional_context=optional_context,
        n_articles=int(n_articles),
    )

    # --- Call Claude
    with st.spinner(f"Generating {int(n_articles)} LinkedIn article(s)…"):
        try:
            md = call_claude_markdown(
                prompt=prompt,
                model=model_id.strip(),
                max_tokens=3200,  # enough headroom for up to 5 articles
                temperature=float(temperature),
            )
        except NotFoundError as e:
            st.error("Claude call failed: model not found.")
            st.markdown(friendly_model_error(model_id.strip()))
            st.exception(e)
            st.stop()
        except AuthenticationError as e:
            st.error("Claude call failed: authentication error (check ANTHROPIC_API_KEY in Streamlit Secrets).")
            st.exception(e)
            st.stop()
        except RateLimitError as e:
            st.error("Claude call failed: rate limit. Try again in a moment.")
            st.exception(e)
            st.stop()
        except APIError as e:
            st.error("Claude call failed: API error.")
            st.exception(e)
            st.stop()
        except Exception as e:
            st.error("Claude call failed. Check your model name and API key in Streamlit Secrets.")
            st.exception(e)
            st.stop()

    # --- Basic output handling
    md = (md or "").strip()
    if not md:
        st.error("Claude returned an empty response. Please try again.")
        st.stop()

    if "```" in md:
        md = strip_code_fences(md)

    # Validate format; if invalid, attempt ONE repair call
    if not looks_like_valid_output(md, int(n_articles)):
        repair_prompt = (
            prompt
            + "\n\nIMPORTANT: Your previous output did not match the required Markdown structure. "
              f"Re-output ONLY in the exact required structure with headings ## Article 1..{int(n_articles)} "
              "and 200–400 words per article. No code fences. No extra text before/after."
        )
        with st.spinner("Repairing output format…"):
            try:
                repaired = call_claude_markdown(
                    prompt=repair_prompt,
                    model=model_id.strip(),
                    max_tokens=3200,
                    temperature=float(temperature),
                )
                repaired = strip_code_fences((repaired or "").strip())
                if repaired:
                    md = repaired
            except Exception:
                pass

    articles, split_ok = split_into_n_articles(md, int(n_articles))

    st.session_state.result_md = md
    st.session_state.articles = articles
    st.session_state.n_articles_last = int(n_articles)

    if not split_ok:
        st.warning(
            "Claude didn’t return the exact expected structure (Article 1..N). "
            "Showing raw output in the Copy section so you can still use it."
        )

# ---------------------------
# Results display
# ---------------------------
if st.session_state.result_md:
    st.markdown("---")
    st.subheader("Results")
    st.info(st.session_state.parse_info)

    n = st.session_state.n_articles_last
    tabs = st.tabs([f"Article {i}" for i in range(1, n + 1)])

    # If parsing failed, we might only have 1 blob
    for i, tab in enumerate(tabs, start=1):
        with tab:
            if i - 1 < len(st.session_state.articles):
                st.markdown(st.session_state.articles[i - 1])
            else:
                st.markdown("_No parsed content for this tab._")

    with st.expander("Copy (Markdown)", expanded=False):
        st.caption("Tip: Use the copy icon in the top-right of the code block.")
        st.code(st.session_state.result_md, language="markdown")

    with st.expander("Troubleshooting", expanded=False):
        st.markdown(
            "- Authentication error: confirm **ANTHROPIC_API_KEY** is set in Streamlit Secrets.\n"
            "- Model-not-found: pick a valid model in **1. Context → Model**.\n"
            "- Too generic: add 1–2 lines of optional context and/or increase max answers sent.\n"
            "- Too similar articles: raise temperature slightly or increase number of answers.\n"
        )
else:
    st.caption("Paste a question + answers, choose number of articles, then click **Generate**.")
