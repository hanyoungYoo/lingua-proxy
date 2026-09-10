# Security policy

## Reporting a vulnerability

Please report security issues privately through
[GitHub's private vulnerability reporting](https://github.com/hanyoungYoo/lingua-proxy/security/advisories/new)
rather than opening a public issue.

## What this software touches

lingua-proxy sits on the path between your LLM client and an API, so it is worth
being explicit about what that means.

- **It sees every prompt and reply.** That is inherent to being a proxy.
- **It forwards your credential.** Whatever `Authorization` or `x-api-key` header
  your client sends is passed to the upstream unchanged, and reused for the
  translation call unless you set a separate `translator.api_key`.
- **It stores prompts on disk.** The translation memo at
  `~/.lingua-proxy/memo.jsonl` holds your prompts and replies in plaintext, mode
  `600`. Set `memo.persist = false` to keep it in memory only. Do not attach this
  file to a bug report.
- **It binds to loopback only.** The proxy would otherwise be an open relay for
  your API key. Do not expose it to a network you do not control.
- **The cost log holds counts, never text**, so it is safe to share.

## Scope

In scope: credential leakage, prompt content leaking somewhere it should not go,
request smuggling or header injection through the proxy, and any path that sends
data to an endpoint the user did not configure.

Out of scope: the security of upstream APIs, and the inherent fact that a proxy
can read the traffic you route through it.
