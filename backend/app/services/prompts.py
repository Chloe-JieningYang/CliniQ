"""Role-specific system prompts and message construction (DPO train/serve aligned).

DPO training prepends a fixed self-role line at the start of each example's prompt
(`I am a doctor.` / `I am a patient.`). The API mirrors that in the user turn so
inference matches the distribution seen during preference optimization.
"""

from typing import Any, List, Literal, Optional

Role = Literal["doctor", "patient"]

_SYSTEM_DOCTOR = (
    "You are CliniQ, a medical knowledge assistant speaking to a licensed healthcare "
    "professional. Use precise clinical terminology where appropriate, cite mechanisms "
    "and guideline-style reasoning when useful, and mention relevant differential "
    "considerations or follow-up when clinically warranted. Do not substitute for "
    "in-person evaluation or institutional protocols."
)

_SYSTEM_PATIENT = (
    "You are CliniQ, a medical information assistant speaking to a lay patient. "
    "Explain in clear, everyday language; define unavoidable medical terms briefly; "
    "emphasize when to seek urgent or in-person care; avoid alarming or definitive "
    "diagnoses. This is educational information only, not personal medical advice."
)

# When RAG or client supplies context, steer away from copying case studies / vignettes.
_CONTEXT_USE_INSTRUCTIONS = (
    "Use the reference material below.\n"
    "Step 1: Normalize it—rewrite as general medical knowledge only; remove specific people, "
    "patients, or studies; resolve pronouns (e.g., he, this study) into neutral clinical facts; "
    "keep symptoms, causes, mechanisms, treatments, and outcomes; omit case-specific narrative.\n"
    "Step 2: Using only that normalized knowledge, answer my question above. Do not mention "
    "patients, studies, or cases as stories; give a concise, clinically relevant answer.\n\n"
    "Context:\n{ctx}"
)


def dpo_user_role_prefix(role: Role) -> str:
    """Leading line used in DPO dataset prompts (must match training)."""
    if role == "doctor":
        return "I am a doctor."
    return "I am a patient."


def system_prompt_for_role(role: Role) -> str:
    """Return the system instruction for the selected audience."""
    if role == "doctor":
        return _SYSTEM_DOCTOR
    return _SYSTEM_PATIENT


def build_user_content(role: Role, message: str, context: Optional[str]) -> str:
    """Build user text: DPO role prefix, question, then optional context with de-vignette steps."""
    prefix = dpo_user_role_prefix(role)
    body = message.strip()
    if context and context.strip():
        ctx = context.strip()
        instructions = _CONTEXT_USE_INSTRUCTIONS.format(ctx=ctx)
        return f"{prefix}\n{body}\n\n{instructions}"
    return f"{prefix}\n{body}"


def stub_chat_answer(role: Role, message: str, context: Optional[str]) -> str:
    """Fixed reply for CLINIQ_MOCK_GENERATION — no model call."""
    role_label = "doctor" if role == "doctor" else "patient"
    preview = message.strip()[:120]
    ctx_line = ""
    if context and context.strip():
        ctx_line = f"\nContext preview: {context.strip()[:120]}"
    return (
        "[Mock mode] No LLM weights are loaded; this is a fixed placeholder for API/UI integration.\n"
        f"Audience: {role_label} (DPO-style user prefix: {dpo_user_role_prefix(role)}).\n"
        f"Question preview: {preview}{ctx_line}\n"
        "Load real adapters and set CLINIQ_MOCK_GENERATION=false to see model-generated text here."
    )


def build_chat_messages(
    role: Role,
    message: str,
    context: Optional[str],
) -> List[dict[str, Any]]:
    """Messages list for `tokenizer.apply_chat_template` (Llama 3 Instruct)."""
    return [
        {"role": "system", "content": system_prompt_for_role(role)},
        {"role": "user", "content": build_user_content(role, message, context)},
    ]
