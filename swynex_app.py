"""
SWYNEX - Intelligent AI Assistant (single-file version)
Final AI Application | SWYNEX Technologies AI Internship

Stack: Python | Google Gemini 2.5 Flash | Google GenAI SDK | Streamlit | JSON | unit tests

Setup:
    pip install streamlit google-genai python-dotenv
    set GEMINI_API_KEY=your_key        (Windows)   /   export GEMINI_API_KEY=your_key  (Mac/Linux)
    streamlit run swynex_app.py

Run unit tests (no API key needed):
    python swynex_app.py --test
"""
import json
import os
import re
import sys
import unittest
from dataclasses import asdict, dataclass
from typing import Optional

# ============================================================ CONFIG
MODEL_NAME = "gemini-2.5-flash"
MIN_INPUT_CHARS, MAX_INPUT_CHARS = 20, 20000
MIN_TOPIC_CHARS, MAX_TOPIC_CHARS = 3, 300
SUMMARY_STYLES = ["Paragraph", "Bullet points", "One-line TL;DR"]
SUMMARY_LENGTHS = ["Short", "Medium", "Detailed"]
CONTENT_TYPES = ["Blog post", "Email", "Social media post", "Product description", "Essay", "Story"]
TONES = ["Professional", "Casual", "Friendly", "Persuasive", "Formal"]
CONTENT_LENGTHS = ["Short", "Medium", "Long"]
SCORE_MIN, SCORE_MAX = 1, 10
CRITERIA = ("relevance", "completeness", "clarity")

SYSTEM_PROMPT = ("You are SWYNEX, a helpful, accurate and responsible AI assistant. "
                 "Do not invent facts, stay faithful to the user's input, and be clear and well organised.")
EVAL_SYSTEM_PROMPT = ("You are a strict, fair evaluator of AI responses. Score each criterion from 1 to 10 "
                      "(10 = excellent). Respond with JSON only, no extra text.")


# ============================================================ ERRORS
class SwynexError(Exception):
    """Base class for application errors."""


class ValidationError(SwynexError):
    """User input failed validation."""


class AIServiceError(SwynexError):
    """Gemini failed or returned unusable output."""


# ============================================================ VALIDATION
def validate_text(text, field="Text", min_len=MIN_INPUT_CHARS, max_len=MAX_INPUT_CHARS) -> str:
    if not isinstance(text, str):
        raise ValidationError(f"{field} must be text.")
    cleaned = text.strip()
    if not cleaned:
        raise ValidationError(f"{field} cannot be empty.")
    if len(cleaned) < min_len:
        raise ValidationError(f"{field} is too short (minimum {min_len} characters).")
    if len(cleaned) > max_len:
        raise ValidationError(f"{field} is too long (maximum {max_len} characters).")
    return cleaned


def validate_topic(topic) -> str:
    return validate_text(topic, "Topic", MIN_TOPIC_CHARS, MAX_TOPIC_CHARS)


def validate_choice(value, allowed, field) -> str:
    if value not in allowed:
        raise ValidationError(f"{field} must be one of: {', '.join(allowed)}.")
    return value


def validate_api_key(key) -> str:
    if not isinstance(key, str) or not key.strip():
        raise ValidationError("A Gemini API key is required. Add it in the sidebar or set GEMINI_API_KEY.")
    key = key.strip()
    if len(key) < 20 or " " in key:
        raise ValidationError("That API key doesn't look valid.")
    return key


# ============================================================ GEMINI CLIENT
class GeminiClient:
    def __init__(self, api_key: str, model: str = MODEL_NAME, client=None):
        self.model = model
        if client is not None:  # injected for tests
            self._client = client
        else:
            from google import genai
            self._client = genai.Client(api_key=api_key)

    def generate(self, prompt: str, system_instruction: Optional[str] = None,
                 temperature: float = 0.7, json_output: bool = False) -> str:
        try:
            from google.genai import types
            config = types.GenerateContentConfig(
                system_instruction=system_instruction, temperature=temperature,
                response_mime_type="application/json" if json_output else "text/plain")
            response = self._client.models.generate_content(
                model=self.model, contents=prompt, config=config)
            text = (response.text or "").strip()
        except Exception as exc:
            raise AIServiceError(self._friendly(exc)) from exc
        if not text:
            raise AIServiceError("The model returned an empty response. Try rephrasing your input.")
        return text

    @staticmethod
    def _friendly(exc: Exception) -> str:
        msg = str(exc)
        low = msg.lower()
        if any(k in low for k in ("api key", "api_key", "permission", "401", "403")):
            return "Authentication failed. Please check your Gemini API key."
        if any(k in low for k in ("429", "quota", "rate", "resource_exhausted")):
            return "Rate limit or quota reached. Please wait a moment and try again."
        if any(k in low for k in ("timeout", "deadline", "unavailable", "503")):
            return "The AI service is temporarily unavailable. Please try again shortly."
        if "safety" in low or "blocked" in low:
            return "The request was blocked by safety filters. Please rephrase your input."
        return f"AI request failed: {msg[:200]}"


# ============================================================ FEATURES
_LENGTH_HINT = {"Short": "about 2-3 sentences",
                "Medium": "about one short paragraph (4-6 sentences)",
                "Detailed": "a thorough summary covering all key points"}
_CONTENT_HINT = {"Short": "around 100 words", "Medium": "around 250 words", "Long": "around 450 words"}


def build_summary_prompt(text, style, length) -> str:
    fmt = {"Bullet points": "Use concise bullet points.",
           "One-line TL;DR": "Write exactly one sentence."}.get(style, "Write in flowing paragraph form.")
    size = "" if style == "One-line TL;DR" else f"Length: {_LENGTH_HINT[length]}. "
    return f"Summarize the text below. {fmt} {size}Only use information from the text.\n\nTEXT:\n{text}"


def summarize(client, text, style="Paragraph", length="Medium") -> str:
    text = validate_text(text, "Text to summarize")
    style = validate_choice(style, SUMMARY_STYLES, "Style")
    length = validate_choice(length, SUMMARY_LENGTHS, "Length")
    return client.generate(build_summary_prompt(text, style, length),
                           system_instruction=SYSTEM_PROMPT, temperature=0.3)


def generate_content(client, topic, content_type="Blog post", tone="Professional", length="Medium") -> str:
    topic = validate_topic(topic)
    content_type = validate_choice(content_type, CONTENT_TYPES, "Content type")
    tone = validate_choice(tone, TONES, "Tone")
    length = validate_choice(length, CONTENT_LENGTHS, "Length")
    prompt = (f"Write a {content_type.lower()} about: {topic}\n"
              f"Tone: {tone.lower()}. Length: {_CONTENT_HINT[length]}.\n"
              "Make it engaging, well structured and factually careful.")
    return client.generate(prompt, system_instruction=SYSTEM_PROMPT, temperature=0.8)


# ============================================================ EVALUATOR
@dataclass
class Evaluation:
    relevance: int
    completeness: int
    clarity: int
    feedback: str

    @property
    def overall(self) -> float:
        return round((self.relevance + self.completeness + self.clarity) / 3, 1)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["overall"] = self.overall
        return d


def _clamp(value) -> int:
    try:
        number = int(round(float(value)))
    except (TypeError, ValueError):
        raise AIServiceError("Evaluation returned a non-numeric score.")
    return max(SCORE_MIN, min(SCORE_MAX, number))


def parse_evaluation(raw: str) -> Evaluation:
    """Parse model JSON (tolerates code fences / extra text) into an Evaluation."""
    cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            raise AIServiceError("Could not read the evaluation result.")
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise AIServiceError("Could not read the evaluation result.") from exc
    if not isinstance(data, dict):
        raise AIServiceError("Evaluation result had an unexpected format.")
    missing = [c for c in CRITERIA if c not in data]
    if missing:
        raise AIServiceError(f"Evaluation is missing: {', '.join(missing)}.")
    return Evaluation(_clamp(data["relevance"]), _clamp(data["completeness"]), _clamp(data["clarity"]),
                      str(data.get("feedback", "")).strip() or "No feedback provided.")


def evaluate_response(client, task, user_input, response) -> Evaluation:
    user_input = validate_text(user_input, "Original input", min_len=3, max_len=30000)
    response = validate_text(response, "AI response", min_len=3, max_len=30000)
    prompt = (f"Task type: {task}\n\nUSER INPUT:\n{user_input}\n\nAI RESPONSE:\n{response}\n\n"
              "Evaluate the AI response and return JSON in exactly this shape:\n"
              '{"relevance": <1-10>, "completeness": <1-10>, "clarity": <1-10>, "feedback": "<2 sentences>"}\n'
              "relevance = how well it addresses the input; completeness = covers all key points; "
              "clarity = easy to read and well organised.")
    raw = client.generate(prompt, system_instruction=EVAL_SYSTEM_PROMPT, temperature=0.0, json_output=True)
    return parse_evaluation(raw)


# ============================================================ UNIT TESTS
class _Fake:
    def __init__(self, reply="ok"):
        self.reply, self.calls = reply, []

    def generate(self, prompt, system_instruction=None, temperature=0.7, json_output=False):
        self.calls.append(prompt)
        return self.reply


class _SDK:
    class _M:
        def __init__(s, r): s.r = r
        def generate_content(s, **kw):
            if isinstance(s.r, Exception):
                raise s.r
            return s.r
    def __init__(self, r): self.models = self._M(r)


class _Resp:
    def __init__(self, t): self.text = t


_GOOD = json.dumps({"relevance": 9, "completeness": 8, "clarity": 7, "feedback": "Solid."})
_LONG = "Artificial intelligence is transforming how software is built. " * 3


class SwynexTests(unittest.TestCase):
    def test_validate_text(self):
        self.assertEqual(validate_text("  " + "a" * 30 + " "), "a" * 30)
        for bad in ["", "   ", None, 5, "short", "a" * 20001]:
            with self.assertRaises(ValidationError):
                validate_text(bad)

    def test_validate_topic_choice_key(self):
        self.assertEqual(validate_topic("IoT"), "IoT")
        with self.assertRaises(ValidationError):
            validate_topic("ab")
        with self.assertRaises(ValidationError):
            validate_choice("X", ["A"], "F")
        self.assertEqual(validate_api_key("k" * 25), "k" * 25)
        for bad in ["", None, "short", "a b " + "k" * 30]:
            with self.assertRaises(ValidationError):
                validate_api_key(bad)

    def test_summarize(self):
        c = _Fake("summary")
        self.assertEqual(summarize(c, _LONG, "Bullet points", "Short"), "summary")
        self.assertIn("bullet", c.calls[0].lower())
        with self.assertRaises(ValidationError):
            summarize(_Fake(), "tiny")
        with self.assertRaises(ValidationError):
            summarize(_Fake(), _LONG, style="Haiku")
        self.assertIn("exactly one sentence", build_summary_prompt(_LONG, "One-line TL;DR", "Short"))

    def test_generate_content(self):
        c = _Fake("post")
        self.assertEqual(generate_content(c, "Smart farming", "Email", "Friendly", "Short"), "post")
        with self.assertRaises(ValidationError):
            generate_content(_Fake(), "")
        with self.assertRaises(ValidationError):
            generate_content(_Fake(), "Topic here", tone="Angry")

    def test_parse_evaluation(self):
        ev = parse_evaluation(_GOOD)
        self.assertEqual((ev.relevance, ev.completeness, ev.clarity, ev.overall), (9, 8, 7, 8.0))
        self.assertEqual(parse_evaluation("```json\n" + _GOOD + "\n```").clarity, 7)
        self.assertEqual(parse_evaluation("Here: " + _GOOD + " thanks").relevance, 9)
        clamped = parse_evaluation(json.dumps({"relevance": 99, "completeness": 5, "clarity": -4}))
        self.assertEqual((clamped.relevance, clamped.clarity), (10, 1))
        for raw in ["not json", "[]", json.dumps({"relevance": 5}),
                    json.dumps({"relevance": "x", "completeness": 1, "clarity": 1})]:
            with self.assertRaises(AIServiceError):
                parse_evaluation(raw)

    def test_evaluate_response(self):
        self.assertEqual(evaluate_response(_Fake(_GOOD), "Sum", "input text", "answer").to_dict()["overall"], 8.0)
        with self.assertRaises(ValidationError):
            evaluate_response(_Fake(_GOOD), "x", "", "answer")

    def test_gemini_client(self):
        self.assertEqual(GeminiClient("k" * 25, client=_SDK(_Resp("  hi "))).generate("p"), "hi")
        with self.assertRaises(AIServiceError):
            GeminiClient("k" * 25, client=_SDK(_Resp(""))).generate("p")
        for err, expect in [("API key not valid", "Authentication"),
                            ("429 quota", "Rate limit"), ("503 unavailable", "temporarily")]:
            with self.assertRaisesRegex(AIServiceError, expect):
                GeminiClient("k" * 25, client=_SDK(Exception(err))).generate("p")


# ============================================================ STREAMLIT UI
def main_ui():
    import streamlit as st
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    st.set_page_config(page_title="SWYNEX - Intelligent AI Assistant", page_icon="🧠", layout="wide")

    def default_api_key() -> str:
        key = os.getenv("GEMINI_API_KEY", "")
        if not key:
            try:
                key = st.secrets.get("GEMINI_API_KEY", "")
            except Exception:
                key = ""
        return key

    def show_evaluation(ev):
        st.markdown("#### 📊 Response quality")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Relevance", f"{ev.relevance}/10")
        c2.metric("Completeness", f"{ev.completeness}/10")
        c3.metric("Clarity", f"{ev.clarity}/10")
        c4.metric("Overall", f"{ev.overall}/10")
        st.progress(min(ev.overall / 10, 1.0))
        st.info(ev.feedback)

    def run_task(key, label, user_input, fn, api_key, auto_eval):
        try:
            client = GeminiClient(validate_api_key(api_key))
            with st.spinner("Gemini is thinking..."):
                output = fn(client)
            evaluation = None
            if auto_eval:
                with st.spinner("Evaluating response quality..."):
                    try:
                        evaluation = evaluate_response(client, label, user_input, output)
                    except SwynexError as exc:
                        st.warning(f"Evaluation skipped: {exc}")
            st.session_state[key] = {"output": output, "eval": evaluation}
            st.session_state.history.insert(0, {"task": label, "input": user_input[:120], "output": output})
        except SwynexError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Unexpected error: {exc}")

    def render_result(key):
        result = st.session_state.get(key)
        if result:
            st.markdown("#### ✅ Result")
            st.write(result["output"])
            st.download_button("⬇️ Download as .txt", result["output"], file_name=f"{key}.txt", key=f"dl_{key}")
            if result["eval"]:
                show_evaluation(result["eval"])

    st.session_state.setdefault("history", [])

    with st.sidebar:
        st.title("🧠 SWYNEX")
        st.caption("Intelligent AI Assistant")
        api_key = st.text_input("Gemini API key", value=default_api_key(), type="password",
                                help="Free key from Google AI Studio. Kept only for this session.")
        auto_eval = st.toggle("Auto-evaluate responses", value=True,
                              help="Scores each answer for relevance, completeness and clarity.")
        st.caption(f"Model: `{MODEL_NAME}`")
        if st.button("🗑️ Clear history"):
            st.session_state.history = []
            for k in ("summary", "content", "evaluation"):
                st.session_state.pop(k, None)
            st.rerun()
        st.divider()
        st.markdown("**Responsible AI**: AI output can contain mistakes. Always verify important information.")

    st.title("SWYNEX – Intelligent AI Assistant")
    st.write("Summarize text, generate content, and evaluate the quality of AI responses.")
    tab_sum, tab_gen, tab_eval, tab_hist = st.tabs(
        ["📝 Summarize", "✍️ Generate content", "🔍 Evaluate a response", "🕘 History"])

    with tab_sum:
        text = st.text_area("Paste the text to summarize", height=220, key="sum_text")
        c1, c2 = st.columns(2)
        style = c1.selectbox("Style", SUMMARY_STYLES, key="sum_style")
        length = c2.selectbox("Length", SUMMARY_LENGTHS, index=1, key="sum_len")
        st.caption(f"{len(text)} characters")
        if st.button("Summarize", type="primary"):
            run_task("summary", "Summarization", text, lambda c: summarize(c, text, style, length),
                     api_key, auto_eval)
        render_result("summary")

    with tab_gen:
        topic = st.text_input("Topic", placeholder="e.g. Benefits of IoT in smart agriculture", key="gen_topic")
        c1, c2, c3 = st.columns(3)
        ctype = c1.selectbox("Content type", CONTENT_TYPES, key="gen_type")
        tone = c2.selectbox("Tone", TONES, key="gen_tone")
        glen = c3.selectbox("Length", CONTENT_LENGTHS, index=1, key="gen_len")
        if st.button("Generate", type="primary"):
            run_task("content", "Content generation", topic,
                     lambda c: generate_content(c, topic, ctype, tone, glen), api_key, auto_eval)
        render_result("content")

    with tab_eval:
        st.write("Paste any prompt and an AI-written answer to score it.")
        e_in = st.text_area("Original prompt / input", height=120, key="eval_in")
        e_out = st.text_area("AI response to evaluate", height=160, key="eval_out")
        if st.button("Evaluate", type="primary"):
            try:
                client = GeminiClient(validate_api_key(api_key))
                with st.spinner("Evaluating..."):
                    st.session_state["evaluation"] = evaluate_response(client, "General", e_in, e_out)
            except SwynexError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Unexpected error: {exc}")
        if st.session_state.get("evaluation"):
            show_evaluation(st.session_state["evaluation"])

    with tab_hist:
        if not st.session_state.history:
            st.info("Nothing here yet. Run a task and it will appear in this list.")
        for item in st.session_state.history[:20]:
            with st.expander(f"{item['task']} – {item['input']}"):
                st.write(item["output"])


# ============================================================ ENTRY POINT
if __name__ == "__main__" and "--test" in sys.argv:
    sys.argv.remove("--test")
    unittest.main(verbosity=2)
else:
    main_ui()
