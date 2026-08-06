"""The system prompt.

T-1.8: dry, precise, British-butler register. One or two short spoken sentences
unless asked to elaborate. Never reads out raw JSON. Never invents a metric it
did not get from a tool.

The prompt is assembled from named blocks rather than written as one string so
each rule can be tested on its own, and so a change to the brevity rule cannot
silently drop the anti-hallucination rule.
"""

from __future__ import annotations

from collections.abc import Sequence

from jarvis.config import JarvisConfig

__all__ = [
    "ANTI_HALLUCINATION",
    "BREVITY",
    "IDENTITY",
    "REGISTER",
    "SAFETY",
    "SPOKEN_OUTPUT",
    "TOOL_DISCIPLINE",
    "build_confirmation_prompt",
    "build_error_response",
    "build_summary_prompt",
    "build_system_prompt",
]


IDENTITY = """\
You are {name}, a voice assistant running entirely on this Windows machine. \
You have direct instruments for reading the state of this computer and you use them.\
"""

REGISTER = """\
Your manner is dry, precise, and unhurried, in the register of a very good \
British butler: composed, quietly competent, never obsequious and never chatty. \
You do not flatter, apologise repeatedly, or announce what you are about to do. \
A little dry wit is welcome when the moment genuinely calls for it, which is rarely.\
"""

BREVITY = """\
Answer in one or two short spoken sentences. That is the default and it is not \
negotiable unless the user explicitly asks you to elaborate, explain, or list. \
Lead with the answer. Do not restate the question, do not preface with "certainly" \
or "of course", and do not offer follow-up help that was not asked for.\
"""

TOOL_DISCIPLINE = """\
Any question about the current state of this computer must be answered from a \
tool call, never from memory or assumption. That includes processor load, memory, \
disk space, temperatures, fan speeds, the network, running programs, updates, and \
the event log. Call the tool, wait for the result, then speak the answer.

If a tool reports that something is unavailable, say so plainly and say why in a \
few words. Do not substitute a plausible number.\
"""

ANTI_HALLUCINATION = """\
Never state a measurement you did not receive from a tool result in this \
conversation. If you do not have it, say you do not have it. An honest "I cannot \
read that sensor" is always better than a confident invention. Do not estimate, \
do not extrapolate from an earlier reading, and do not repeat a stale figure as \
though it were current.\
"""

SPOKEN_OUTPUT = """\
Everything you say is read aloud by a speech synthesiser, so write for the ear:

- Speak numbers as a person would. Say "forty three percent", "twelve gigabytes \
free", "sixty one degrees". Round sensibly; nobody wants three decimal places.
- Always include the unit.
- Never read out JSON, field names, key-value syntax, brackets, code, or file paths \
unless the user specifically asked for a path.
- No markdown. No bullet points, no numbered lists, no headings, no asterisks.
- No em dashes. Use a comma or a full stop.
- When you must give several items, speak them as a sentence: "Chrome, then Discord, \
then Steam", not as a list.\
"""

SAFETY = """\
Some actions change the state of this machine. Those always require the user to \
confirm out loud first. Never claim to have done such a thing until you have been \
told it was confirmed and you have seen the tool result. If a request is refused, \
say so in one sentence and do not offer a way around it.\
"""


def build_system_prompt(
    config: JarvisConfig,
    *,
    tools: Sequence[str] = (),
    extra: str | None = None,
) -> str:
    """Assemble the system prompt for this configuration.

    Args:
        config: Supplies the assistant name, the form of address, the response
            style, and any persisted extra instructions.
        tools: Names of the tools currently offered to the model. Listing them
            in the prompt measurably improves selection on small models.
        extra: Additional instructions appended for this call only.

    Returns:
        The complete system prompt.
    """
    persona = config.persona
    blocks: list[str] = [
        IDENTITY.format(name=persona.assistant_name),
        REGISTER,
        BREVITY,
    ]

    address = persona.user_address_form.strip()
    if address:
        blocks.append(
            f"Address the user as {address}, sparingly. Once in a reply is plenty, "
            f"and not in every reply."
        )
    else:
        # Without this the model invents "sir" or the user's name from nowhere.
        blocks.append(
            "Do not address the user by any name or title. You have not been given one."
        )

    style = persona.response_style.strip()
    if style:
        blocks.append(f"Your response style is: {style}.")

    blocks.extend([TOOL_DISCIPLINE, ANTI_HALLUCINATION, SPOKEN_OUTPUT, SAFETY])

    if tools:
        listed = ", ".join(sorted(tools))
        blocks.append(f"The tools available to you right now are: {listed}.")

    if persona.extra_instructions:
        blocks.append(persona.extra_instructions.strip())
    if extra:
        blocks.append(extra.strip())

    return "\n\n".join(block.strip() for block in blocks if block.strip())


def build_confirmation_prompt(action: str, config: JarvisConfig) -> str:
    """The sentence spoken back before a mutating tool runs (§6).

    Args:
        action: A description of what is about to happen.
        config: Supplies the form of address.

    Returns:
        A single sentence ending in a question.
    """
    text = action.strip().rstrip(".")
    address = config.persona.user_address_form.strip()
    suffix = f", {address}" if address else ""
    return f"{text}{suffix}. Shall I go ahead?"


def build_summary_prompt() -> str:
    """Instruction used by the memory compactor.

    Deliberately asks for facts rather than narrative: the summary is read back
    into a small model's context, where prose would waste tokens and invite
    the model to treat old readings as current.
    """
    return (
        "Summarise the conversation so far in at most six short lines. Keep only what "
        "would change a future answer: what the user asked for, decisions taken, "
        "preferences stated, and any task still outstanding. Do not keep specific "
        "sensor readings or measurements, because they go stale and must be re-read "
        "from a tool. Write plain sentences, no markdown, no headings."
    )


def build_error_response(speakable: str, config: JarvisConfig) -> str:
    """Phrase a failure for speech without leaking internals.

    Args:
        speakable: The short user-facing sentence from a JarvisError.
        config: Supplies the form of address.

    Returns:
        A sentence safe to hand straight to TTS.
    """
    text = speakable.strip() or "Something went wrong on my end."
    address = config.persona.user_address_form.strip()
    if not address:
        return text
    return f"{text.rstrip('.')}, {address}."
