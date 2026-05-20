# Security Policy

Maracatu is an open-weight LLM research project. We take both code security and model risk concerns (privacy, bias, dangerous content generation) seriously. Thank you for helping us keep the project safe and responsible.

## Reporting a vulnerability or model issue

**Do not open public issues for vulnerabilities.** Instead:

- Use [GitHub Security Advisories](https://github.com/maracatu-labs/maracatu/security/advisories/new) (preferred — private by default), **or**
- Send an email to **contact@maracatu.org** with:
  - Description of the issue
  - Steps to reproduce
  - Potential impact
  - Suggested mitigation, if you have one

We will confirm receipt within 72 hours and work with you to understand and fix the issue. After the fix is published, we may credit you in the release note (if you wish).

## Scope

### Code (traditional security)

- Training pipeline (`src/maracatu/`)
- Corpus preparation and deploy scripts (`scripts/`)
- Configs (`configs/`)
- Tokenizer (`tokenizer/`)

Examples: code injection via unsanitized input, unsafe checkpoint deserialization, credential leakage in logs.

### Model (responsible safety)

- Generation of clearly dangerous content (instructions for violence, self-harm, etc.)
- Detectable leakage of training corpus data (memorization)
- Severe systematic bias (racism, sexism, extremist content) that exceeds the expected baseline of a pretrained model without instruction tuning

**Important:** Maracatu is a base model (no RLHF / instruction tuning until the 800M release). Behaviors typical of a base model — repetition, inconsistent generation, lack of filters — are not vulnerabilities. We document this scope in [MODEL_CARD.md](MODEL_CARD.md).

### Out of scope

- Vulnerabilities in third-party dependencies (PyTorch, transformers, datasets) — report to the original maintainer first. Let us know if it affects Maracatu directly.
- Unexpected outputs that fall within the expected behavior of a PT-BR base model.
- Attacks that rely on physical access or social engineering against specific contributors.

## Best practices for users of the weights

- **Do not use Maracatu in critical production without appropriate fine-tuning and a safety layer.** The released models are pretrained (base models), without alignment.
- Audit outputs before exposing them to end users.
- Consider quantization and local inference to reduce attack surface (leakage of sensitive inputs to external APIs).
- For commercial use, read the [Apache 2.0](LICENSE) — you can, but you are responsible for the use.

## Advisory history

When we publish a security advisory, it will appear under [GitHub Security Advisories](https://github.com/maracatu-labs/maracatu/security/advisories).
