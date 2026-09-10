# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project uses
[semantic versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-09-10

First release. Alpha: the interfaces may still change.

### Added

- Translating reverse proxy for the Anthropic Messages API
  (`POST /v1/messages`) and the OpenAI-compatible chat API
  (`POST /v1/chat/completions`). Everything else passes through untouched.
- Offline language detection with `lingua`. English input is a pure passthrough
  with no added latency and no extra tokens.
- Technical-content masking, so fenced code, inline code, URLs, file paths,
  JSON blobs and markup tags survive translation byte for byte.
- Bidirectional translation memo, in memory and on disk, which keeps resent
  conversation history byte-stable so the upstream prompt cache keeps hitting.
- Streaming support. Text blocks are buffered per block and translated; tool
  calls and reasoning stream through untouched, with keep-alives while a
  translation is in flight.
- `wrap` and `unwrap` commands that patch and transactionally restore client
  settings, chaining onto whatever gateway the client already points at.
- `doctor`, `dashboard`, and `bench` commands. The benchmark reports savings in
  dollars and is able to report that a workload does not pay off.
- `x-lingua-bypass` header to skip translation for a single request.

### Known limitations

- Savings apply to input tokens. Output tokens cost more and are not reduced,
  so end-to-end savings are lower than the input-token figure.
- Streaming buffers each text block, so non-English text does not appear word
  by word.
- macOS and Linux only.
- DeepL and LibreTranslate adapters are interface-only.

[Unreleased]: https://github.com/hanyoungYoo/lingua-proxy/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/hanyoungYoo/lingua-proxy/releases/tag/v0.1.0
