"""Role-specific system prompts and message construction."""

from typing import Any, List, Literal, Optional

Role = Literal["practitioner", "patient"]

_SYSTEM_PRACTITIONER = (
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


def system_prompt_for_role(role: Role) -> str:
    """Return the system instruction for the selected audience."""
    if role == "practitioner":
        return _SYSTEM_PRACTITIONER
    return _SYSTEM_PATIENT


def build_user_content(message: str, context: Optional[str]) -> str:
    """Combine optional context with the user question (matches SFT-style user block)."""
    message = message.strip()
    if context and context.strip():
        return f"{message}\nContext: {context.strip()}"
    return message


def stub_chat_answer(role: Role, message: str, context: Optional[str]) -> str:
    """Fixed reply for CLINIQ_MOCK_GENERATION — no model call."""
    role_label = "clinician" if role == "practitioner" else "patient"
    preview = message.strip()[:120]
    ctx_line = ""
    if context and context.strip():
        ctx_line = f"\nContext preview: {context.strip()[:120]}"
    return (
        "[Mock mode] No LLM weights are loaded; this is a fixed placeholder for API/UI integration.\n"
        f"Audience: {role_label}.\n"
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
        {"role": "user", "content": build_user_content(message, context)},
    ]
