import re
from typing import List, Tuple, Dict, Optional

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


def looks_like_valid_output(md: str) -> bool:
    if not md or not md.strip():
        return False
    if "```" in md:
        return False
    needed = [r"^##\s*Article\s*1\b", r"^##\s*Article\s*2\b", r"^##\s*Article\s*3\b"]
    for pat in needed:
        if not re.search(pat, md, flags=re.MULTILINE | re.IGNORECASE):
            return False
    return True


def strip_code_fences(md: str) -> str:
    fenced = re.match(r"^\s*```(?:\w+)?\s*(.*?)\s*```\s*$", md, flags=re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    md = re.sub(r"(?m)^\s*```.*?$", "", md)
    return md.strip()


def split_into_three_articles(md: str) -> Tuple[str, str, str, bool]:
    pattern = r"(?m)^##\s*Article\s*([123])\b.*$"
    matches = list(re.finditer(pattern, md, flags=re.IGNORECASE | re.MULTILINE))
    if len(matches) < 3:
        return md.strip(), "", "", False

    starts: Dict[str, int] = {}
    for m in matches:
        num = m.group(1)
        if num not in starts:
            starts[num] = m.start()

    if not all(k in starts for k in ("1", "2", "3")):
        return md.strip(), "", "", False

    ordered = sorted(((k, v) for k, v in starts.items()), key=lambda x: x[1])
    slices = {}
    for i, (num, start) in enumerate(ordered):
        end = ordered[i + 1][1] if i + 1 < len(ordered) else len(md)
        slices[num] = md[start:end].strip()

    return slices["1"], slices["2"], slices["3"], True


def build_prompt(question: str, answers: List[str], optional_context: str, company_name: str) -> str:
    answers_block = "\n".join([f"{i+1}. {a}" for i, a in enumerate(answers)])
    ctx = (optional_context or "").strip()
    ctx_block = ctx if ctx else "No extra context provided."

    return f"""
You are a senior thought-leadership writer specialising in the forest industry and its value chain (pulp, paper, packaging, hygiene, timber, printing/publishing, brand owners, converters, logistics, procurement, sustainability, innovation).

TASK
Write THREE distinct LinkedIn articles based on patterns found in real interview answers.

CONFIDENTIALITY (CRITICAL)
- Do NOT mention any client, brand, mill, retailer, or company names.
- Do NOT reference specific projects, geographies that could identify a single account, or “one specific customer”.
- Do NOT write “our client told us X”.
- Use aggregated language such as “Across many conversations…” / “A recurring pattern is…”.
- Keep it concrete enough to feel real, but abstract enough to protect confidentiality.
- Avoid quoting verbatim sentences from the input.

POSITIONING (SUBTLE)
Without being salesy, subtly position {company_name} as:
- close to customers’ customers across the forest industry value chain,
- able to see patterns others miss,
- focused on strategic, forward-looking issues.
You MAY mention {company_name} by name lightly (max 1–2 times total across all articles), but the authority must come from insight, not promotion.

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

## Article 1 — <Angle title>
<200–400 words>

## Article 2 — <Angle title>
<200–400 words>

## Article 3 — <Angle title>
<200–400 words>

STYLE (LinkedIn-ready)
- Strong opening hook (1–2 short paragraphs).
- Scannable formatting: short paragraphs; optional bullet list (max 4 bullets) if helpful.
- Thoughtful, reflective, credible; written for senior decision-makers in the forest industry.
- Clear takeaway at the end (1–2 sentences).
- Do not use a sales CTA (“contact us”, “book a call”, etc.).

QUALITY BAR
Each article must have a distinct:
- angle,
- narrative arc (hook → insight → implications → takeaway),
- emphasis (choose different themes across the three).
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
    "Turn raw interview answers into **three distinct, insight-driven LinkedIn articles** — "
    "credible, anonymised, and designed for senior decision-makers in the forest industry."
)

MODEL_PRESETS = {
    "Claude Sonnet 4.5 — stable snapshot (recommended)": "claude-sonnet-4-5-20250929",
    "Claude Sonnet 4.5 — alias": "claude-sonnet-4-5",
    "Claude Haiku 4.5 — alias (fast/cheap)": "claude-haiku-4-5",
    "Claude Opus 4.5 — alias (premium)": "claude-opus-4-5",
    "Custom (type your own model ID)": "__custom__",
}

with st.expander("1. Context", expanded=True):
    st.caption("Optional framing — keep it light. The app will still work if you leave this empty.")
    company_name = st.text_input("Company name (used subtly in the articles)", value="Opticom")

    optional_context = st.text_area(
        "Optional context (e.g., what this question was about, what theme you want to emphasise, what audience nuance to consider)",
        placeholder="Example: These answers come from conversations across the value chain about supply reliability, credibility of claims, and procurement trade-offs.",
        height=120,
    )

    col_a, col_b, col_c = st.columns([1.2, 1, 1])
    with col_a:
        model_choice = st.selectbox("Model", list(MODEL_PRESETS.keys()), index=0)
        custom_model = ""
        if MODEL_PRESETS[model_choice] == "__custom__":
            custom_model = st.text_input("Custom model ID", placeholder="e.g. claude-sonnet-4-5-20250929")

        model_id = custom_model.strip() if custom_model.strip() else MODEL_PRESETS[model_choice]
        if model_id == "__custom__":
            model_id = ""  # force validation below

        st.caption(
            "If you previously used `claude-3-5-sonnet-latest`, it will 404 (model ID no longer exists). "
            "Use one of the options above."
        )

    with col_b:
        max_answers_to_send = st.number_input("Max answers to send to Claude", min_value=20, max_value=500, value=160, step=10)
    with col_c:
        temperature = st.slider("Creativity (temperature)", min_value=0.0, max_value=1.0, value=0.6, step=0.05)

with st.expander("2. Paste Input", expanded=True):
    st.caption("Paste directly from Excel. Each row can have multiple columns; the app will pick the longest cell per row as the answer.")
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
    st.session_state.article_1 = ""
    st.session_state.article_2 = ""
    st.session_state.article_3 = ""
    st.session_state.parse_info = ""

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
        company_name=company_name.strip() or "Opticom",
    )

    # --- Call Claude
    with st.spinner("Generating three LinkedIn articles…"):
        try:
            md = call_claude_markdown(
                prompt=prompt,
                model=model_id.strip(),
                max_tokens=2500,
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
    if not looks_like_valid_output(md):
        repair_prompt = (
            prompt
            + "\n\nIMPORTANT: Your previous output did not match the required Markdown structure. "
              "Re-output ONLY in the exact required structure with three headings (## Article 1/2/3) "
              "and 200–400 words per article. No code fences."
        )
        with st.spinner("Repairing output format…"):
            try:
                repaired = call_claude_markdown(
                    prompt=repair_prompt,
                    model=model_id.strip(),
                    max_tokens=2500,
                    temperature=float(temperature),
                )
                repaired = strip_code_fences((repaired or "").strip())
                if repaired:
                    md = repaired
            except Exception:
                pass

    a1, a2, a3, split_ok = split_into_three_articles(md)

    st.session_state.result_md = md
    st.session_state.article_1 = a1
    st.session_state.article_2 = a2
    st.session_state.article_3 = a3

    if not split_ok:
        st.warning(
            "Claude did not return the exact expected structure (3 headings). "
            "Showing the raw output in the Copy section so you can still use it."
        )

# ---------------------------
# Results display
# ---------------------------
if st.session_state.result_md:
    st.markdown("---")
    st.subheader("Results")
    st.info(st.session_state.parse_info)

    tab1, tab2, tab3 = st.tabs(["Article 1", "Article 2", "Article 3"])

    with tab1:
        st.markdown(st.session_state.article_1 or "_No Article 1 parsed._")
    with tab2:
        st.markdown(st.session_state.article_2 or "_No Article 2 parsed._")
    with tab3:
        st.markdown(st.session_state.article_3 or "_No Article 3 parsed._")

    with st.expander("Copy (Markdown)", expanded=False):
        st.caption("Tip: Use the copy icon in the top-right of the code block.")
        st.code(st.session_state.result_md, language="markdown")

    with st.expander("Troubleshooting", expanded=False):
        st.markdown(
            "- If you see an authentication error: confirm **ANTHROPIC_API_KEY** is set in Streamlit Secrets.\n"
            "- If you see a model-not-found error: pick a valid model in **1. Context → Model**.\n"
            "- If outputs feel too generic: add 1–2 lines of optional context and/or increase max answers sent.\n"
        )
else:
    st.caption("Paste a question + answers, then click **Generate**.")
